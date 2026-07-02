from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Boolean,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

if False:
    @dataclass
    class RequestFuncOutput:
        """The output of the request function including metrics."""

        generated_text: str = ""
        success: bool = False
        latency: float = 0.0
        output_tokens: int = 0
        ttft: float = 0.0  # Time to first token
        itl: list[float] = field(default_factory=list)  # list of inter-token latencies
        tpot: float = 0.0  # avg next-token latencies
        prompt_len: int = 0
        error: str = ""
        start_time: float = 0.0


@dataclass
class TimelineConfig:
    num_buckets: int | None = 30
    bucket_duration: float | None = None

    input_token_weight: float = 1.0
    output_token_weight: float = 5.0

    latency_percentiles: tuple[float, ...] = (0.5, 0.90, 0.95, 0.99)
    track_latency: bool = True


# ---------------------------------------------------------------------------
#  SQLAlchemy persistence — store raw results once, replay with different configs
# ---------------------------------------------------------------------------

_Base = declarative_base()


class _Run(_Base):
    __tablename__ = "runs"

    run_id      = Column(Integer, primary_key=True, autoincrement=True)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String(512))
    config_json = Column(JSON)

    requests = relationship("_Request", back_populates="run", cascade="all, delete-orphan")


class _Request(_Base):
    __tablename__ = "requests"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(Integer, ForeignKey("runs.run_id"), nullable=False, index=True)
    start_time     = Column(Float)
    latency        = Column(Float)
    prompt_len     = Column(Integer)
    output_tokens  = Column(Integer)
    ttft           = Column(Float)
    tpot           = Column(Float)
    itl_json       = Column(JSON)
    success        = Column(Boolean)
    error          = Column(Text)
    generated_text = Column(Text)

    run = relationship("_Run", back_populates="requests")


def _default_db_path() -> Path:
    runs_dir = os.environ.get("MILABENCH_DIR_RUNS")
    if runs_dir:
        return Path(runs_dir) / "benchmark_results.db"
    return Path("benchmark_results.db")


class ResultStore:
    """Single entry-point for benchmark result persistence."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = _default_db_path()
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        self.engine = create_engine(url)
        _Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save(
        self,
        outputs: list[dict],
        description: str = "",
        config: TimelineConfig | None = None,
    ) -> int:
        """Persist raw RequestFuncOutput dicts. Returns the new run_id."""
        config_json = asdict(config) if config else None

        run = _Run(description=description, config_json=config_json)
        for o in outputs:
            if not isinstance(o, dict):
                o = asdict(o)
            run.requests.append(_Request(
                start_time     = o.get("start_time"),
                latency        = o.get("latency"),
                prompt_len     = o.get("prompt_len"),
                output_tokens  = o.get("output_tokens"),
                ttft           = o.get("ttft"),
                tpot           = o.get("tpot"),
                itl_json       = o.get("itl", []),
                success        = bool(o.get("success", False)),
                error          = o.get("error", ""),
                generated_text = o.get("generated_text", ""),
            ))

        with self.Session() as session:
            session.add(run)
            session.commit()
            return run.run_id

    def load(self, run_id: int | None = None) -> list[dict]:
        """Load results for a run. Loads the latest run when run_id is None."""
        with self.Session() as session:
            if run_id is None:
                run_id = session.query(func.max(_Run.run_id)).scalar()
                if run_id is None:
                    return []

            rows = (
                session.query(_Request)
                .filter(_Request.run_id == run_id)
                .order_by(_Request.start_time)
                .all()
            )
            return [
                {
                    "start_time":     r.start_time,
                    "latency":        r.latency,
                    "prompt_len":     r.prompt_len,
                    "output_tokens":  r.output_tokens,
                    "ttft":           r.ttft,
                    "tpot":           r.tpot,
                    "itl":            r.itl_json if r.itl_json else [],
                    "success":        bool(r.success),
                    "error":          r.error or "",
                    "generated_text": r.generated_text or "",
                }
                for r in rows
            ]

    def list_runs(self) -> list[dict]:
        with self.Session() as session:
            rows = (
                session.query(
                    _Run.run_id,
                    _Run.created_at,
                    _Run.description,
                    func.count(_Request.id).label("num_requests"),
                )
                .outerjoin(_Request)
                .group_by(_Run.run_id)
                .order_by(_Run.run_id)
                .all()
            )
            return [
                {
                    "run_id": r.run_id,
                    "created_at": str(r.created_at),
                    "description": r.description or "",
                    "num_requests": r.num_requests,
                }
                for r in rows
            ]


# ---------------------------------------------------------------------------
#  Bucket-count heuristic
# ---------------------------------------------------------------------------

# Minimum completions per bucket for a percentile to be stable.
# p99 of 128 samples is just the max — useless.  You need multiple
# waves of C completions per bucket for extreme percentiles.
SAMPLES_FOR_PERCENTILE = {
    0.50: 20,
    0.90: 50,
    0.95: 100,
    0.99: 500,
}


def _min_samples_for(percentile: float) -> int:
    """How many completions a bucket needs for a given percentile."""
    for p in sorted(SAMPLES_FOR_PERCENTILE):
        if percentile <= p:
            return SAMPLES_FOR_PERCENTILE[p]
    return SAMPLES_FOR_PERCENTILE[0.99]


def suggest_num_buckets(
    num_requests: int,
    concurrency: int = 1,
    highest_percentile: float = 0.99,
    min_buckets: int = 10,
    max_buckets: int = 200,
) -> int:
    """Suggest a bucket count given request count and concurrency.

    Requests complete in waves of ~concurrency.  Each bucket needs
    enough completions for the requested percentile to be stable:

        p50   — ~20  completions/bucket  (~0.2 waves @ C=128)
        p90   — ~50  completions/bucket  (~0.4 waves @ C=128)
        p99   — ~500 completions/bucket  (~4   waves @ C=128)

    With C=128 and N=3000 that is 23 waves total:
        p50 → up to 115 buckets  (23 / 0.2)
        p99 → up to   5 buckets  (23 / 4)

    Note: ITL is denser (output_tokens values per job) so its
    percentiles are more stable than TTFT/TPOT/E2E at the same
    bucket count.
    """
    min_per_bucket = _min_samples_for(highest_percentile)
    ideal = num_requests // min_per_bucket
    return max(min_buckets, min(max_buckets, ideal))


def suggest_num_requests(
    num_buckets: int = 30,
    concurrency: int = 1,
    highest_percentile: float = 0.99,
) -> dict:
    """How many requests the benchmark should send.

    Returns a dict with the minimum request count needed and the
    number of waves (batches of C concurrent requests) that implies.

    Example with C=128, B=30, targeting p99:
        min_per_bucket = 500
        num_requests   = 30 * 500 = 15000
        num_waves      = 15000 / 128 ≈ 117
    """
    min_per_bucket = _min_samples_for(highest_percentile)
    total = num_buckets * min_per_bucket
    num_waves = max(1, total // concurrency) if concurrency else total

    return {
        "num_requests": total,
        "num_waves": num_waves,
        "completions_per_bucket": min_per_bucket,
        "concurrency": concurrency,
        "num_buckets": num_buckets,
        "highest_percentile": highest_percentile,
    }


@dataclass
class Job:
    start: float
    end: float


class Timeline:
    def __init__(self, jobs: list[Job]):
        self.jobs: list[Job] = jobs 

    def pending(self, start, end=None):
        actively_running = []

        for job in self.jobs:
            if job.start <= start and job.end > start:
                actively_running.append(job)
        
        return job


@dataclass
class Worker:
    worker_id: int
    active_job = None
    job_count: int = 0

    def set_job(self, job):
        self.active_job = job

        if job is not None:
            self.job_count += 1
            self.active_job.worker = self.worker_id
            self.active_job.batch_id = self.job_count

    def end(self):
        if self.active_job:
            return self.active_job.end
        return 0


class JobAdapter:
    def __init__(self, dat):
        self.data = dat
        self.start = self.data["start_time"]
        self.end = self.start + self.data["latency"]
        self.pct = None
        self.worker = None
        self.accounted = 0
        self.batch_id = None

    def completion_percentage(self, start, end):
        total_time = self.end - self.start

        work_time_included_in_sampling = min(end, self.end) - max(start, self.start)
        
        percentage_done = max(min(work_time_included_in_sampling / total_time, 0), 1)
        return percentage_done

    def total_token(self, config: TimelineConfig | None = None):
        if config is None:
            total = self.data['output_tokens'] + self.data['prompt_len']
        else:
            total = (
                self.data['prompt_len'] * config.input_token_weight
                + self.data['output_tokens'] * config.output_token_weight
            ) / (config.input_token_weight + config.output_token_weight)

        if self.pct:
            return total * self.pct

        return total

    def token_per_second(self, config: TimelineConfig | None = None):
        if config is None:
            total = self.data['output_tokens'] + self.data['prompt_len']
        else:
            total = (
                self.data['prompt_len'] * config.input_token_weight
                + self.data['output_tokens'] * config.output_token_weight
            ) / (config.input_token_weight + config.output_token_weight)

        elapsed = self.data["latency"]
        return total / elapsed

    def __repr__(self):
        data = self.__json__()
        data.pop("generated_text", None)
        args = ", ".join([f"{k}={v}" for k, v in data.items()])
        return f"Job({args})"

    def __json__(self):
        return {
            **self.data,
            "start": self.start,
            "end": self.end,
            "worker": self.worker,
            "batch_id": self.batch_id
        }

def convert(obj):
    if isinstance(obj, dict):
        return obj
    return asdict(obj)


@dataclass
class PartialJob:
    job: None
    pct: float


def _percentile_sorted(sorted_values, pct):
    """Compute a single percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    idx = pct * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _latency_stats(values: list[float], percentiles: tuple[float, ...]) -> dict:
    if not values:
        return {}
    sv = sorted(values)
    out = {
        "mean": statistics.mean(sv),
        "min": sv[0],
        "max": sv[-1],
    }
    for p in percentiles:
        out[f"p{int(p * 100)}"] = _percentile_sorted(sv, p)
    return out


@dataclass
class Bucket:
    start: float
    end: float
    jobs: list[PartialJob] = None
    tokens: int = 0

    def overlap(self, job):
        return max(0, min(job.end, self.end) - max(job.start, self.start))

    def active_jobs(self):
        return len(self.jobs)

    def active_jobs_pct(self):
        acc = 0
        bucket_duration = self.end - self.start
        for partial in self.jobs:
            acc += self.overlap(partial.job) / bucket_duration
        return acc

    def has_started_in_bucket(self, job):
        return self.start <= job.start <= self.end
    
    def has_finished_in_bucket(self, job):
        return self.start <= job.end <= self.end

    def ran_in_bucket(self, job):
        return job.start < self.start and job.end > self.end

    def _for_each(self, cond):
        acc = 0
        for partial in self.jobs:
            if cond(partial.job):
                acc += 1
        return acc

    def start_job_count(self):
        return self._for_each(self.has_started_in_bucket)

    def finished_job_count(self):
        return self._for_each(self.has_finished_in_bucket)
    
    def ran_through_job_count(self):
        return self._for_each(self.ran_in_bucket)

    def latency_summary(self, percentiles: tuple[float, ...]) -> dict:
        """Aggregate latency metrics based on when each event occurs.

        - TTFT: included if the first token arrived inside this bucket.
        - ITL:  each inter-token latency is included if the token it
          produced arrived inside this bucket.
        """
        ttfts = []
        itls = []

        for partial in self.jobs:
            job = partial.job
            data = job.data

            ttft = data.get("ttft", 0)
            first_token_time = job.start + ttft

            if ttft and self.start <= first_token_time <= self.end:
                ttfts.append(ttft)

            itl_list = data.get("itl")
            if itl_list:
                t = first_token_time
                for gap in itl_list:
                    t += gap
                    if t > self.end:
                        break
                    if t >= self.start:
                        itls.append(gap)

        result = {}
        for name, vals in (("ttft", ttfts), ("itl", itls)):
            stats = _latency_stats(vals, percentiles)
            for stat_name, v in stats.items():
                result[f"{name}_{stat_name}"] = v
        return result


class TimelineProcessor:
    def __init__(self, config: TimelineConfig | None = None):
        self.config = config or TimelineConfig()
        self.finished_jobs = []
        self.start = 0
        self.end = 0
        self.k = 0
        self.total = 0
        self.step = 0
        self.output = []
        self.avg = 0
        self.avg_instant = 0

    def _make_buckets(self):
        """Build buckets from config: either fixed count or fixed duration."""
        cfg = self.config
        duration = self.end - self.start

        if cfg.bucket_duration is not None:
            num = max(1, int(duration / cfg.bucket_duration))
            step = cfg.bucket_duration
        else:
            num = cfg.num_buckets or 30
            step = duration / num

        self.step = step
        self.samples = [self.start + step * (i + 1) for i in range(num)]
        return num

    def __call__(self, outputs: list[RequestFuncOutput], number=None):
        if number is not None:
            self.config.num_buckets = number

        jobs = [JobAdapter(convert(l)) for l in outputs]
        jobs.sort(key=lambda item: item.start)

        self.save_normalized_data(jobs)
    
        return self.method_2(jobs)

    def save_normalized_data(self, jobs, db_path=None):
        outputs = [j.data for j in jobs]
        store = ResultStore(db_path)
        store.save(outputs, config=self.config)

    def method_2(self, jobs: list[Job]):
        cfg = self.config
        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        number = self._make_buckets()

        buckets = []
        for i in range(number):
            buckets.append(Bucket(i * self.step, (i + 1) * self.step, []))

        for job in jobs:
            for bucket in buckets:
                if job.start <= bucket.end and job.end >= bucket.start:
                    raw_total = job.end - job.start

                    if raw_total == 0:
                        print("Malformed job: ", job)

                    total = max(raw_total, 0.001)
                    overlap = max(0, min(job.end, bucket.end) - max(job.start, bucket.start))
                    bucket.jobs.append(PartialJob(job, overlap/total))

        for bucket in buckets:
            for partial in bucket.jobs:
                bucket.tokens += partial.job.total_token(cfg) * partial.pct
                partial.job.accounted += partial.pct

        for job in jobs:
            if job.accounted < 0.9999999:
                print("WARNING: Unaccounted job", job.accounted, job.start, job.end)

        self.output = []
        self.avg = 0
        for bucket in buckets:
            rate = bucket.tokens / (bucket.end - bucket.start)
            self.avg += rate
            entry = {
                "time": bucket.end + self.start,
                "rate": rate,
                "active_jobs": bucket.active_jobs(),
                "start_job": bucket.start_job_count(),
                "finished_job": bucket.finished_job_count(),
                "ran_through": bucket.ran_through_job_count(),
                "active_jobs_pct": bucket.active_jobs_pct(),
            }
            if cfg.track_latency:
                entry.update(bucket.latency_summary(cfg.latency_percentiles))
            self.output.append(entry)
        
        return self.output

    def _on_time_change(self, now, workers):
        cfg = self.config
        while self.k < len(self.samples) and self.samples[self.k] <= now:
            start = self.step * self.k
            end   = start + self.step

            token = 0
            elapsed = self.step

            self.finished_jobs.sort(key=lambda job: job.end)

            while self.finished_jobs and self.finished_jobs[0].end <= end:
                job = self.finished_jobs.pop(0)
                token += job.total_token(cfg)
                job.accounted = True

            instant = 0
            for w in workers:
                if w.active_job:
                    instant += w.active_job.token_per_second(cfg)

            throughput = token / elapsed if elapsed > 0 else 0

            self.avg += throughput
            self.avg_instant += instant

            self.output.append({
                "rate": throughput,
                "time": self.samples[self.k],
                "instant": instant,
            })

            self.k += 1

    def _on_job_ended(self, job, workers):
        if job is None:
            end_time = self.end
        else:
            end_time = job.end
            self.finished_jobs.append(job)

        self._on_time_change(end_time, workers)

    def method_1(self, jobs: list[RequestFuncOutput]):
        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        self._make_buckets()
    
        workers = []
        for job in jobs:
            # Sort by workers tht will finish first
            workers.sort(key=lambda w: w.end())

            for worker in workers:
                # worker is going to finish 
                if worker.end() < job.start:
                    jb = worker.active_job
                    self._on_job_ended(jb, workers)
                    worker.set_job(job)
                    break

            else:
                w = Worker(len(workers))
                workers.append(w)
                w.set_job(job)

        workers.sort(key=lambda w: w.end())
        for worker in workers:
            jb = worker.active_job
            self._on_job_ended(jb, workers)
            worker.set_job(None)
        
        self._on_job_ended(None, workers)
            

        for job in jobs:
            if job.accounted < 0.999:
                print("MISSING")

        return jobs


def timeline(outputs, number=None, config: TimelineConfig | None = None):
    if config is None:
        config = TimelineConfig()
    if number is not None:
        config.num_buckets = number

    proc = TimelineProcessor(config)
    proc(outputs)
    return proc.output


def plot_timeline(jobs):
    import altair as alt

    base = alt.Chart(jobs).encode(
        y=alt.Y(
            "worker:O",
            title="Request",
            sort="-x"
        ),
        x=alt.X(
            "start:Q",
            title="Time (s)",
            axis=alt.Axis(format=".2f"),
        ),
        color="batch_id:O",
        x2="end:Q",
    )

    bars = base.mark_bar(height=12).properties(
        width=900,
        height=25 * len(set(jobs["worker"])),
        title="Request Timeline (Gantt)"
    )
    bars.save('chart.png', scale_factor=2)
    return bars


def main():
    from argparse import ArgumentParser
    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument("file", type=str, nargs="?", default=None,
                        help="JSON file with raw outputs (or .db for SQLite)")
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite database path to load results from")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run ID to load from the database (latest if omitted)")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all runs in the database and exit")
    parser.add_argument("-n", "--num-buckets", type=int, default=None)
    parser.add_argument("--auto-buckets", action="store_true",
                        help="Automatically choose bucket count based on request count and concurrency")
    parser.add_argument("--concurrency", "-C", type=int, default=1,
                        help="Max concurrent requests (used by --auto-buckets)")
    parser.add_argument("--highest-percentile", type=float, default=0.99,
                        help="Target percentile for --auto-buckets (default 0.99)")
    parser.add_argument("--suggest-requests", action="store_true",
                        help="Print how many requests are needed for the given bucket/concurrency/percentile and exit")
    parser.add_argument("--bucket-duration", type=float, default=None,
                        help="Fixed bucket width in seconds (overrides -n)")
    parser.add_argument("--input-weight", type=float, default=1.0,
                        help="Weight applied to input/prompt tokens")
    parser.add_argument("--output-weight", type=float, default=5.0,
                        help="Weight applied to output/generated tokens")
    parser.add_argument("--no-latency", action="store_true",
                        help="Disable per-bucket latency stats")
    
    args = parser.parse_args()

    if args.suggest_requests:
        info = suggest_num_requests(
            num_buckets=args.num_buckets or 30,
            concurrency=args.concurrency,
            highest_percentile=args.highest_percentile,
        )
        print(f"  To get stable p{int(info['highest_percentile']*100)} "
              f"with {info['num_buckets']} buckets and C={info['concurrency']}:")
        print(f"    requests needed : {info['num_requests']}")
        print(f"    waves           : {info['num_waves']}")
        print(f"    completions/bucket: {info['completions_per_bucket']}")
        return

    if args.list_runs:
        db = args.db or args.file
        store = ResultStore(db)
        for run in store.list_runs():
            print(f"  run {run['run_id']:>4d}  {run['created_at']}  "
                  f"{run['num_requests']:>6d} requests  {run.get('description', '')}")
        return

    if args.db or (args.file and args.file.endswith(".db")):
        db = args.db or args.file
        store = ResultStore(db)
        outputs = store.load(run_id=args.run_id)
    elif args.file:
        with open(args.file, "r") as fp:
            outputs = json.load(fp)
    else:
        parser.error("Provide a JSON file, --db path, or --list-runs")

    num_buckets = args.num_buckets
    if num_buckets is None:
        if args.auto_buckets:
            num_buckets = suggest_num_buckets(
                len(outputs),
                concurrency=args.concurrency,
                highest_percentile=args.highest_percentile,
            )
            print(f"Auto-selected {num_buckets} buckets for "
                  f"{len(outputs)} requests (C={args.concurrency}, "
                  f"p{int(args.highest_percentile*100)})")
        else:
            num_buckets = 30

    config = TimelineConfig(
        num_buckets=num_buckets,
        bucket_duration=args.bucket_duration,
        input_token_weight=args.input_weight,
        output_token_weight=args.output_weight,
        track_latency=not args.no_latency,
    )

    proc = TimelineProcessor(config)

    jobs = [JobAdapter(convert(l)) for l in outputs]

    jobs.sort(key=lambda item: item.end)
    jobs.sort(key=lambda item: item.start)

    results = proc.method_2(jobs)

    for line in results:
        print(line)

    _ = proc.method_1(jobs)

    data = pd.DataFrame([job.__json__() for job in jobs])
    print(data)
    plot_timeline(data)


if __name__ == "__main__":
    main()