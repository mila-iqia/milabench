from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict

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
class Job:
    start: float
    end: float


class Timeline:
    def __init__(self, jobs: list[Jobs]):
        self.jobs: list[Jobs] = jobs 

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

    def total_token(self):
        total = (self.data['output_tokens'] + self.data['prompt_len'])

        if self.pct:
            return total * self.pct

        return total

    def token_per_second(self):
        total = (self.data['output_tokens'] + self.data['prompt_len'])
        elapsed = self.data["latency"]
        return total / elapsed

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
        # 1 if the job run through the entire bucket
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


class TimelineProcessor:
    def __init__(self):
        self.finished_jobs = []
        self.start = 0
        self.end = 0
        self.k = 0
        self.total = 0
        self.step = 0
        self.output = []
        self.avg = 0
        self.avg_instant = 0

    def _sample(self, number):
        self.step = (self.end - self.start) / (number)
        self.samples = [self.start + self.step * (i + 1) for i in range(number)]

    def __call__(self, outputs: list[RequestFuncOutput], number=30):
        jobs = [JobAdapter(convert(l)) for l in outputs]
        jobs.sort(key=lambda item: item.start)

        self.save_normalized_data(jobs)
    
        return self.method_2(jobs, number)

    def save_normalized_data(self, jobs):
        if True:
            import time
            import json

            with open(f"fjobs_{int(time.time())}.json", "w") as fp:    
                json.dump([j.data for j in jobs], fp)

    def method_2(self, jobs: list[jobs], number=30):
        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        self._sample(number)

        buckets = []
        for i in range(number):
            buckets.append(Bucket(i * self.step, (i + 1) * self.step, []))

        for job in jobs:
            for bucket in buckets:
                if job.start <= bucket.end and job.end >= bucket.start:
                    total = job.end - job.start
                    overlap = max(0, min(job.end, bucket.end) - max(job.start, bucket.start))
                    bucket.jobs.append(PartialJob(job, overlap/total))

        for bucket in buckets:
            for partial in bucket.jobs:
                bucket.tokens += partial.job.total_token() * partial.pct
                partial.job.accounted += partial.pct

        for job in jobs:
            if job.accounted < 0.9999999:
                print("WARNING: Unaccounted job", job.accounted, job.start, job.end)

        self.output = []
        self.avg = 0
        for bucket in buckets:
            rate = bucket.tokens / (bucket.end - bucket.start)
            self.avg += rate
            self.output.append({
                "time": bucket.end,
                "rate": rate,
                "active_jobs": bucket.active_jobs(),
                "start_job": bucket.start_job_count(),
                "finished_job": bucket.finished_job_count(),
                "ran_through": bucket.ran_through_job_count(),
                "active_jobs_pct": bucket.active_jobs_pct(),
            })
        
        return self.output

    def _on_time_change(self, now, workers):
        while self.k < len(self.samples) and self.samples[self.k] <= now:
            start = self.step * self.k
            end   = start + self.step

            token = 0
            elapsed = self.step

            # Sort by job end time
            self.finished_jobs.sort(key=lambda job: job.end)

            # Consume jobs that ended inside this bucket
            while self.finished_jobs and self.finished_jobs[0].end <= end:
                job = self.finished_jobs.pop(0)
                token += job.total_token()
                job.accounted = True

            # Instant throughput (running jobs)
            instant = 0
            for w in workers:
                if w.active_job:
                    instant += w.active_job.token_per_second()

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

    def method_1(self, jobs: list[RequestFuncOutput], number):
        start = jobs[0].start
        for job in jobs:
            job.start -= start
            job.end -= start

            self.start = min(job.start, self.start)
            self.end = max(job.end, self.end)

        self._sample(number)
    
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


def timeline(outputs, number):
    proc = TimelineProcessor()
    proc(outputs, number)
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
    import json
    from argparse import ArgumentParser
    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-n", type=int, default=30)
    
    args = parser.parse_args()

    with open(args.file, "r") as fp:
        outputs = json.load(fp)

    proc = TimelineProcessor()

    #
    # Generate the global rate
    #
    jobs = [JobAdapter(convert(l)) for l in outputs]

    jobs.sort(key=lambda item: item.end)
    jobs.sort(key=lambda item: item.start)

    results = proc.method_2(jobs, args.n)

    for line in results:
        print(line)

    #
    # Plot the Jobs
    #
    _ = proc.method_1(jobs, args.n)

    data = pd.DataFrame([job.__json__() for job in jobs])
    print(data)
    plot_timeline(data)


if __name__ == "__main__":
    main()