from __future__ import annotations

from contextlib import contextmanager
import mimetypes
import os
import subprocess
import json
import re
import tempfile
import os
import yaml
import uuid
from functools import wraps
import traceback
import time
import threading
from filelock import FileLock, Timeout

from cantilever.core.timer import timeit
from flask import request, jsonify, send_file

from .constant import *


#
#   The integration assume the server can send SSH command to the SLURM cluster
#   The server does not have a direct access to the SLURM cluster
#   The server copies file over to the slurm cluster in a know folder structure
#   known as the JOBRUNNER_WORKDIR, this folder has all the job data that matters
#
#   This folder is periodically sync locally to the server and it is used as cache
#   No database needed, all the information is saved on the filesystem and sync between the cluster and the local machine
#
#   /home/$user/scratch/jobrunner/$internal_job_id/log.stdout
#   /home/$user/scratch/jobrunner/$internal_job_id/log.stderr
#   /home/$user/scratch/jobrunner/$internal_job_id/meta/acc.json    <= cached accounting information
#   /home/$user/scratch/jobrunner/$internal_job_id/meta/info.json   <= cached scontrol | squeue  information
#   /home/$user/scratch/jobrunner/$internal_job_id/script.sbatch
#   /home/$user/scratch/jobrunner/$internal_job_id/cmd.sh
#   /home/$user/scratch/jobrunner/$internal_job_id/output/...
#


OR = "?"
AND = ","

AFTER = "after"
AFTER_OK = "afterok"
AFTER_ANY = "afterany"
AFTER_NOT_OK = "afternotok"
SINGLETON = "singleton"


class JobNode:
    def gen(self, depends_event=AFTER_OK, depends_on=None):
        pass

    @staticmethod
    def from_json(data):
        def make_jobs(data):
            return [JobNode.from_json(job) for job in data.pop("jobs")]

        match data.pop("type"):
            case "job":
                return Job(**data)
            case "parallel":
                return Parallel(*make_jobs(data), **data)
            case "sequential":
                return Sequential(*make_jobs(data), **data)
            case "pipeline":
                return Pipeline(job_definition=JobNode.from_json(data.pop("definition")), **data)
            case unknown_type:
                raise RuntimeError(f"Unknown type: {unknown_type}")

class Pipeline:
    """The job runner define a dependency between jobs and schedule them to slurm.
    It is able to launch a job on the cluster.

    Its most interesting feature is being able to track which job failed and which were successful
    and only trigger the failed one.

    To do so it simply iterate over the job tree and mark the one failed to retry and skip the successful ones.

    The pipeline runs alterate the folder structure to make the output fs reflect the job dependencies

    """

    def __init__(self, name, job_definition, job_id=None):
        self.definition: JobNode = job_definition
        self.name = name

        # A pipeline run is something only known to the job runner and as such does not have Slurm Job ID
        # But it has a jobs id
        self.job_id = job_id

    def output_dir(self, root=JOBRUNNER_WORKDIR):
        return os.path.join(root, self.name)

    def schedule(self):
        context = {
            "output": [self.output_dir()]
        }

        self.definition.gen(context)

    def rerun(self) -> Pipeline:
        # traverse the job definition and create a new one
        pass

    def __json__(self):
        return {
            "type": "pipeline",
            "definition": self.definition.__json__(),
            "job_id": self.job_id,
            "name": self.name
        }


class Job(JobNode):
    def __init__(self, script, profile, job_id=None, slurm_jobid=None):
        self.script = script
        self.profile = profile
        self.job_id = job_id
        self.slurm_jobid = slurm_jobid

    def output_dir(self, root=JOBRUNNER_WORKDIR):
        return os.path.join(root, self.job_id)

    def gen(self, context, depends_event=AFTER_OK, depends_on=None):
        sbatch_args = []

        if depends_on is not None:
            sbatch_args.append(f"--dependency={depends_event}:{depends_on}")

        # fetch resource profile
        # ...

        # fetch script
        # ...

        # Create the job folder locally and rsync it to remote

        # Build the sbatch command

        # Submit the job, get a slurm_id for dependencies if any
        return self.slurm_jobid

    def __json__(self):
        return {
            "type": "job",
            "script": self.script,
            "profile": self.profile,
            "job_id": self.job_id,
            "slurm_jobid": self.slurm_jobid
        }


class Sequential(JobNode):
    def __init__(self, *jobs, name='S'):
        self.name = name
        self.job_id = None
        self.jobs = jobs

    def output_dir(self, root):
        return os.path.join(root, self.name)

    def gen(self, depends_on=None):
        job_id = depends_on

        for job in self.jobs:
            job_id = job.gen(depends_on=job_id)

        return job_id

    def __json__(self):
        return {
            "type": "sequential",
            "name": self.name,
            "jobs": [
                job.__json__() for job in self.jobs
            ]
        }


class Parallel(JobNode):
    def __init__(self, *jobs, name='P'):
        self.name = name
        self.jobs = jobs

    def output_dir(self, root):
        return os.path.join(root, self.name)

    def gen(self, depends_on=None):
        job_ids = []
        for job in self.jobs:
            job_ids.append(job.gen(depends_on=depends_on))

        return ':'.join(job_ids)

    def __json__(self):
        return {
            "type": "parallel",
            "name": self.name,
            "jobs": [
                job.__json__() for job in self.jobs
            ]
        }

#
# Standard milabench run
#
standard_run = Sequential(
    Job("pin", "pin"),
    Job("install", "install"),
    Job("prepare", "prepare"),
    Parallel(
        Job("A100l", "run"),
        Job("A100", "run"),
        Job("A6000", "run"),
        Job("H100", "run"),
        Job("L40S", "run"),
        Job("rtx8000", "run"),
        Job("v100", "run"),
    )
)


def is_state_terminal(state):
    return state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY")


def get_acc_state(acc):
    states = acc.get("state", {}).get("current", [])

    if len(states) > 0:
        return states[0]
    return None

def get_info_state(acc):
    states = acc.get("job_state", [])

    if len(states) > 0:
        return states[0]
    return None

def is_acc_terminal(acc):
    if type(acc) is list:
        return False
    state = get_acc_state(acc)
    return (state is not None) and is_state_terminal(state)


def job_acc_cache_status(filename: str):
    with open(filename, "r") as fp:
        try:
            job_acc = json.load(fp)

            return is_acc_terminal(job_acc)
        except json.decoder.JSONDecodeError:
            return False


def local_cache(cache_key, arg_key="job_id", check_validation=None):
    """Cache a remote result locally"""
    def wrapped(fun):
        @wraps(fun)
        def wrapper(**kwargs):
            job_id = kwargs.get(arg_key)

            cache_dir = safe_job_path(job_id, "meta")
            cache_file = os.path.join(cache_dir, cache_key)

            try:
                if os.path.exists(cache_file):
                    if check_validation is None or check_validation(cache_file):
                        return send_file(cache_file, mimetype="application/json")
            except:
                traceback.print_exc()

            result = fun(**kwargs)

            if type(result) is tuple:
                # request failed, send old data
                if os.path.exists(cache_file):
                    return send_file(cache_file, mimetype="application/json")

                # No cached data
                return result

            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "w") as fp:
                    fp.write(result.get_data(as_text=True))
            except AttributeError:
                traceback.print_exc()
                return result

            return result
        return wrapper
    return wrapped


def generate_unique_job_name(job_data=None, from_job_id=None):
    """Try to generate a meaningful name from the job configuration"""
    unique_id = str(uuid.uuid4())[:8]

    if from_job_id is not None:
        return f"{from_job_id[:-9]}_{unique_id}"

    if job_data is not None:

        # If a job name is provided, use that and add a unique identifier
        if job_name := job_data.get('job_name', None):
            return f"{job_name}_{unique_id}"

        # Exract script arguments ?

    return unique_id


RSYNC_OK = 0
RSYNC_TIMEOUT = 2
RSYNC_ERROR = 1

def clean_remote():
    try:
        cmd = f"find {JOBRUNNER_WORKDIR} -type d -mtime +7 -exec rm -rf {{}} +"
        remote_command("mila", cmd, timeout=30)
    except:
        pass


def remove_job_from_remote(jr_job_id):
    assert jr_job_id != ""

    cmd = f"rm -rf {JOBRUNNER_WORKDIR}/{jr_job_id}"
    remote_command("mila", cmd, timeout=30)



@contextmanager
def system_file(jr_job_id, filename, defaults):
    cache_dir = JOBRUNNER_LOCAL_CACHE

    if jr_job_id is not None:
        cache_dir = safe_job_path(jr_job_id)

    sys_folder = os.path.join(cache_dir, ".sys")
    cache_file = os.path.join(sys_folder, filename)
    lock_file = cache_file + ".lock"

    os.makedirs(sys_folder, exist_ok=True)
    try:
        with FileLock(lock_file):
            limiter = defaults

            if os.path.exists(cache_file):
                with open(cache_file, "r") as fp:
                    limiter = json.load(fp)

            yield limiter

            with open(cache_file, "w") as fp:
                json.dump(limiter, fp)
    except Timeout:
        yield False


@contextmanager
def cache_invalidator(jr_job_id, filename, limit=30):
    with system_file(jr_job_id, filename, {"last": 0}) as system:
        now =  time.time()
        last = system.get("last", 0)

        if (now - last) > limit:
            system["last"] = now
            yield True
        else:
            yield False


@contextmanager
def job_rsync_limiter(jr_job_id, limit=30):
    with cache_invalidator(jr_job_id, "limiter.json", limit=30) as r:
        yield r


def rsync_jobrunner_folder(timeout=5, force=False):
    with job_rsync_limiter(None) as can_run:
        if not can_run and not force:
            print("RSYNC [full] skipped")
            return 0

        rsync_cmd = ["rsync", "-az", f"mila:{JOBRUNNER_WORKDIR}/", f"{JOBRUNNER_LOCAL_CACHE}"]

        try:
            with timeit("rsync") as chrono:
                # "ssh -T -c aes128-gcm@openssh.com -o Compression=no -x"
                print("RSYNC [full]")
                _ = subprocess.run(
                        rsync_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        shell=False
                    )

            if chrono.timing.value.avg > 2:
                clean_remote()

            print(f"{' '.join(rsync_cmd)} {chrono.timing.value.avg} s")

            return 0
        except subprocess.TimeoutExpired:
            print(f"{' '.join(rsync_cmd)} timed out after {timeout} s")
            return 1
        except Exception:
            import traceback
            traceback.print_exc()
            return 1


def safe_path(base, *frags):
    job_path = os.path.normpath(os.path.join(base, *frags))

    # Ensure resulting path is within JOBRUNNER_LOCAL_CACHE
    if not job_path.startswith(os.path.abspath(base)):
        raise Exception("Invalid fragements")

    return job_path

def safe_job_path(*frags):
    return safe_path(JOBRUNNER_LOCAL_CACHE, *frags)


def validate_jr_job_id(jr_job_id):
    if jr_job_id == "":
        return False

    safe_job_path(JOBRUNNER_LOCAL_CACHE, jr_job_id)

    return True


def rsync_jobrunner_job(jr_job_id, timeout=5):
    if not validate_jr_job_id(jr_job_id):
        return 1

    # I could check the status to avoid the rsync

    with job_rsync_limiter(jr_job_id) as can_run:
        if not can_run:
            print("RSYNC [partial] skipped")
            return 0

        rsync_cmd = ["rsync", "-az", f"mila:{JOBRUNNER_WORKDIR}/{jr_job_id}", f"{JOBRUNNER_LOCAL_CACHE}"]

        try:
            with timeit("rsync") as chrono:
                print("RSYNC [partial]")
                _ = subprocess.run(
                        rsync_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        shell=False
                    )

            print(f"{' '.join(rsync_cmd)} {chrono.timing.value.avg} s")

            return 0
        except subprocess.TimeoutExpired:
            print(f"{' '.join(rsync_cmd)} timed out after {timeout} s")
            return 1
        except Exception:
            import traceback
            traceback.print_exc()
            return 1



def get_active_jobs():
    result = remote_command("mila", "'squeue --json -u $USER'")

    if not result['success']:
        return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 500

    return json.loads(result['stdout'])["jobs"]


def make_comment(dictionary):
    frags = []
    for k, v in dictionary.items():
        frags.append(f"{k}={v}")
    return ";".join(frags)


def parse_comment(comment, results=None):
    if results is None:
        results = {}

    kv_dict = comment.split(';')

    for kv in kv_dict:
        try:
            k, v = kv.split("=")
            results[k] = v
        except ValueError:
            pass
    return results


def book_keeping():
    """This function runs to ensure the remote has a little jobs as possible

    - Do a full rsync
    - List all the jobs present on remove
    - Fetch sacct data for the finished jobs, save that data locally
    - Delete finished jobs on remote

    """
    print("BOOK KEEPING")
    try:
        if rsync_jobrunner_folder(force=True) == 0:
            #
            results = remote_command("mila", f"ls -1 {JOBRUNNER_WORKDIR}", timeout=30)
            all_jobs = results["stdout"].split('\n')

            active_jobs = get_active_jobs()
            active_jr_job_id = [
                parse_comment(job["comment"]).get("jr_job_id", None) for job in active_jobs
            ]

            for job in all_jobs:
                if job == "":
                    continue

                if job in active_jr_job_id:
                    continue

                remove_job_from_remote(job)
    except Exception:
        traceback.print_exc()



def job_rsync_load(cache_key, arg_key="jr_job_id"):
    """Rsync jobrunner data folder and load the file.

    This is used to big-ish file, instead of getting them from the compute cluster directly
    we rsync the jobrunner folder to only get the missing data and then send the local data over.
    """
    def wrapped(fun):
        @wraps(fun)
        def wrapper(**kwargs):
            # Cols = 80
            # Rows = 80
            # Size TO Fetch: (Cols * Rows * 8)

            job_id = kwargs.get(arg_key)
            start = kwargs.get("start")
            end = kwargs.get("end")

            cache_dir = safe_job_path(job_id)
            cache_file = os.path.join(cache_dir, cache_key)

            status = "rsynced"
            if rsync_jobrunner_job(jr_job_id=job_id) != 0:
                status = "stale"

            if os.path.exists(cache_file):
                with open(cache_file, "r") as fp:
                    if start is not None and end is not None:
                        fp.seek(start, os.SEEK_SET)
                        return {
                            "data": fp.read(end - start),
                            "size": os.path.getsize(cache_file),
                            "status": status
                        }

                    return {
                        "data": fp.read(),
                        "size": os.path.getsize(cache_file),
                        "status": status
                    }
            return {
                "data": "",
                "size": 0,
                "status": "N/A"
            }
            #return fun(**kwargs)
        return wrapper
    return wrapped

def validate_slurm_job_id(s_job_id):
    return int(s_job_id)

def full_rsync_load(cache_key, arg_key="job_id"):
    """Rsync jobrunner data folder and load the file.

    This is used to big-ish file, instead of getting them from the compute cluster directly
    we rsync the jobrunner folder to only get the missing data and then send the local data over.
    """
    def wrapped(fun):
        @wraps(fun)
        def wrapper(**kwargs):
            # Cols = 80
            # Rows = 80
            # Size TO Fetch: (Cols * Rows * 8)

            job_id = kwargs.get(arg_key)
            start = kwargs.get("start")
            end = kwargs.get("end")

            cache_dir = safe_job_path(job_id)
            cache_file = os.path.join(cache_dir, cache_key)

            status = "rsynced"
            if rsync_jobrunner_folder() != 0:
                status = "stale"

            if os.path.exists(cache_file):
                with open(cache_file, "r") as fp:
                    if start is not None and end is not None:
                        fp.seek(start, os.SEEK_SET)
                        return {
                            "data": fp.read(end - start),
                            "size": os.path.getsize(cache_file),
                            "status": status
                        }

                    return {
                        "data": fp.read(),
                        "size": os.path.getsize(cache_file),
                        "status": status
                    }
            return {
                "data": "",
                "size": 0,
                "status": "N/A"
            }
            #return fun(**kwargs)
        return wrapper
    return wrapped


def _parse_sbatch_args(sbatch_args):
    """Parse sbatch arguments into a structured format"""
    parsed = {}

    for arg in sbatch_args:
        if arg.startswith('--'):
            # Handle --key=value or --key value
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key[2:]  # Remove --
            else:
                key = arg[2:]  # Remove --
                value = None

            # Map to form-friendly names
            if key == 'job-name':
                parsed['job_name'] = value
            elif key == 'partition':
                parsed['partition'] = value
            elif key == 'nodes':
                parsed['nodes'] = int(value) if value else 1
            elif key == 'ntasks':
                parsed['ntasks'] = int(value) if value else 1
            elif key == 'cpus-per-task':
                parsed['cpus_per_task'] = int(value) if value else 1
            elif key == 'mem':
                parsed['mem'] = value if value else '8G'
            elif key == 'time':
                parsed['time_limit'] = value if value else '02:00:00'
            elif key == 'gpus-per-task':
                parsed['gpus_per_task'] = value if value else '1'
            elif key == 'ntasks-per-node':
                parsed['ntasks_per_node'] = int(value) if value else 1
            elif key == 'exclusive':
                parsed['exclusive'] = True
            elif key == 'export':
                parsed['export'] = value if value else 'ALL'
        elif arg.startswith('-w'):
            # Handle nodelist (-w option)
            parsed['nodelist'] = arg[3:] if len(arg) > 3 else ''

    return parsed


def remote_command(host, command, timeout=5):
    """Execute a Slurm command via SSH"""
    try:
        # Remote execution via SSH
        # ssh_cmd = ['ssh', SLURM_HOST, command]
        ssh_cmd = f'ssh {host} {command}'

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True
        )

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': -1
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def local_command(*args, timeout=10):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            # shell=True
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def slurm_integration(app, cache):
    """Add Slurm integration routes to the Flask app"""

    book_keeping()
    app.scheduler.add_job(book_keeping, 'interval', seconds=3600)

    # Configuration for SSH connection to Slurm cluster
    SLURM_HOST = os.environ.get('SLURM_HOST', 'mila')

    @app.route('/api/slurm/jobs/persited/old')
    def api_slurm_persisted_old():
        """Get a list of job output still available"""
        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, "ls scratch/jobrunner")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 500

            jobs = list(filter(lambda x: x != "", result['stdout'].split("\n")))

            # Sort by creation time (oldest first) to preserve server send order
            def get_remote_creation_time(job_id):
                stat_result = remote_command(SLURM_HOST, f"stat -c %Y scratch/jobrunner/{job_id} 2>/dev/null || echo '0'")
                if stat_result['success']:
                    try:
                        return float(stat_result['stdout'].strip())
                    except ValueError:
                        return 0
                return 0

            jobs = sorted(jobs, key=get_remote_creation_time)

            return jsonify(jobs)

        except Exception as e:
            traceback.print_exec()
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/persited')
    def api_slurm_persisted():
        return api_slurm_persisted_limited(100)

    @app.route("/api/slurm/status")
    def api_slurm_status():
        # this works even during maintenance
        # nc -z -w 2 login.server.mila.quebec 2222
        result = remote_command(SLURM_HOST, f"sinfo -h")

        if result['success']:
            return {
                "status": "online"
            }

        return {
            "status": "offline",
            "reason": result["stderr"]
        }

    @app.route('/api/slurm/jobs/persited/<int:limit>')
    def api_slurm_persisted_limited(limit=None):
        """Get a list of job output still available"""
        try:
            if rsync_jobrunner_folder() != 0:
                print("Stale")

            # jobs = os.listdir(JOBRUNNER_LOCAL_CACHE)
            # jobs.remove(".git")
            def load_info(dir_path, modification_time):
                key = f"{dir_path}:{modification_time}"

                if value := cache.get(key):
                    return value

                info_path = os.path.join(dir_path, "meta", "info.json")

                if os.path.exists(info_path):
                    with open(info_path, "r") as fp:
                        try:
                            info = json.load(fp)
                            cache.set(key, info, timeout=3600)
                            return info
                        except Exception:
                            return {}
                return {}

            def load_acc(dir_path, jr_job_id, job_id):
                key = dir_path
                def queue_update():
                    if job_id is not None:
                        import threading

                        def update():
                            acc = fetch_latest_job_acc_cached(jr_job_id, job_id)
                            cache.set(key, acc, timeout=3600)

                        threading.Thread(target=update).start()

                if value := cache.get(key):
                    if not is_acc_terminal(value):
                        print("Job acc not terminal, updating")
                        queue_update()
                    return value

                acc_path = os.path.join(dir_path, "meta", "acc.json")


                # Note that the job acc info might be out dated
                # We could check the job status make sure it is a terminal state
                if os.path.exists(acc_path):
                    with open(acc_path, "r") as fp:
                        try:
                            info = json.load(fp)

                            if not is_acc_terminal(info):
                                print("Job acc not terminal, updating")
                                queue_update()
                            else:
                                cache.set(key, info, timeout=3600)
                            return info
                        except Exception:
                            print(acc_path)
                            traceback.print_exc()
                            queue_update()
                            return {}

                if job_id is not None:
                    print("Job acc missing, loading")
                    queue_update()

                return {}

            # Get all job directories and sort by creation time (oldest first) to preserve server send order
            job_dirs = []
            for item in os.listdir(JOBRUNNER_LOCAL_CACHE):
                item_path = safe_job_path(item)
                if os.path.isdir(item_path) and item[0] != '.':
                    stat= os.stat(item_path)
                    info = load_info(item_path, stat.st_mtime)
                    job_dirs.append({
                        "name": item,
                        "creation_time": stat.st_ctime,
                        "last_modified": stat.st_mtime,
                        "last_accessed": stat.st_atime,
                        "freshness": time.time() - stat.st_mtime,
                        "info": info,
                        "acc": load_acc(item_path, item, info.get("job_id"))
                    })

            # Sort by creation time (oldest first) to maintain server send order
            jobs = sorted(job_dirs, key=lambda x: x["creation_time"], reverse=True)

            if limit is None:
                return jobs

            return jobs[:limit]

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/all')
    def api_slurm_all_jobs():
        """Get list of all slurm jobs"""
        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, "'sacct -u $USER --json'")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 404

            jobs = json.loads(result['stdout'])["jobs"]

            return jsonify(jobs)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<int:job_id>')
    def api_slurm_active_job_status(job_id):
        """Get list of all slurm jobs"""
        job_id = validate_slurm_job_id(job_id)

        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, f"'squeue -u $USER -j {job_id} --json'")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 404

            jobs = json.loads(result['stdout'])["jobs"][0]

            return jsonify(jobs)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/old/<int:job_id>')
    def api_slurm_old_job_status(job_id):
        """Get list of all slurm jobs"""
        job_id = validate_slurm_job_id(job_id)

        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, f"'sacct -u $USER -j {job_id} --json'")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 404

            jobs = json.loads(result['stdout'])["jobs"][0]

            return jsonify(jobs)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # List jobs
    @app.route('/api/slurm/jobs')
    def api_slurm_jobs():
        """Get list of pending slurm jobs"""
        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, "'squeue --json -u $USER'")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 500

            jobs = json.loads(result['stdout'])["jobs"]

            # Extract jr_job_id from comment field for each job
            for job in jobs:
                job['jr_job_id'] = None

                if (comment := job.get('comment')) is not None:
                    kv_dict = comment.split(';')

                    for kv in kv_dict:
                        try:
                            k, v = kv.split("=")
                            job[k] = v
                        except ValueError:
                            pass

            return jsonify(jobs)

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/job/save/<string:jr_job_id>/<string:message>')
    def api_slurm_save_job(jr_job_id: str, message: str):
        try:
            if rsync_jobrunner_job(jr_job_id) == 0:
                cache_dir = safe_job_path(jr_job_id)

                cmd = f"git add {cache_dir} && git commit -m \"{message}\" && git push origin main"

                subprocess.check_call(cmd, shell=True, cwd=cache_dir)

            return jsonify({})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

        return 

    
    @app.route('/api/slurm/rerun/<string:jr_job_id>')
    def api_slurm_rerun(jr_job_id: str):
        """Rerun a previous job"""

        old_jr_job_id = jr_job_id
        new_jr_job_id = generate_unique_job_name(from_job_id=jr_job_id)

        old_local_dir = f"{JOBRUNNER_LOCAL_CACHE}/{old_jr_job_id}"
        new_local_dir = f"{JOBRUNNER_LOCAL_CACHE}/{new_jr_job_id}"
        new_remote_dir = f"scratch/jobrunner/{new_jr_job_id}"

        old_local_script = f"{old_local_dir}/script.sbatch"
        new_local_script = f"{new_local_dir}/script.sbatch"

        old_cmd_path = f"{old_local_dir}/cmd.sh"

        # Create remote directory and copy script
        result = local_command("mkdir", "-p", new_local_dir)

        if not result['success']:
            return jsonify({'error': f'Failed to create job directory: {result["stderr"]}'}), 500

        # Copy old_remote_script to new job
        result = local_command('cp', old_local_script, new_local_script)
        if not result['success']:
            return jsonify({'error': f'Failed to copy script: {result["stderr"]}'}), 500

        result = local_command('cat', old_cmd_path)
        if not result['success']:
            return jsonify({'error': f'Failed to retrieve previous job command: {result["stderr"]}'}), 500

        old_cmd = result["stdout"]

        # Remove old dependency requirements
        old_cmd = re.sub(r'--dependency=[a-z:0-9,\?]*', '', old_cmd)

        new_cmd = "'" + old_cmd.replace(old_jr_job_id, new_jr_job_id) + "'"

        with open(f"{new_local_dir}/cmd.sh", "w") as fp:
            fp.write(new_cmd[1:-1])
            fp.flush()

        # rsync the job to remote
        result = local_command("rsync", "-az", new_local_dir + "/", f"{SLURM_HOST}:{new_remote_dir}")
        if not result['success']:
            return jsonify({'error': f'Failed to retrieve previous job command: {result["stderr"]}'}), 500

        # Submit job
        result = remote_command(SLURM_HOST, new_cmd)

        if result['success']:
            # Extract job ID from output
            job_id_match = re.search(r'Submitted batch job (\d+)', result['stdout'])
            job_id = job_id_match.group(1) if job_id_match else None

            return jsonify({
                'success': True,
                'job_id': job_id,
                "jr_job_id": new_jr_job_id,
                'message': result['stdout']
            })
        else:
            return jsonify({'error': result['stderr']}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/earlysync/<job_id>')
    def api_early_sync(jr_job_id, job_id):
        # This can happen quite frequently because it is (compute node -> local)
        squeue_info = safe_job_path(jr_job_id, "meta", "info.json")
        if os.path.exists(squeue_info):
            with open(squeue_info, "r") as fp:
                info = json.load(fp)

            host = info.get("batch_host")
            compute_node = f"{host}.server.mila.quebec"

            result = local_command(
                "rsync", "-az", f"{compute_node}:/tmp/{job_id}/cuda/results/runs/", f"{JOBRUNNER_LOCAL_CACHE}/{jr_job_id}/runs"
            )

            if result['success']:
                return {"status": "ok"}

        return {"status": "notok"}


    # Submit job
    @app.route('/api/slurm/submit', methods=['POST'])
    def api_slurm_submit():
        """Submit a new Slurm job"""
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400


            jr_job_id = generate_unique_job_name(data)

            # Create a temporary SLURM script
            job_name = data.get('job_name', jr_job_id)

            script_content = data.get('script', '')
            remote_dir = f"~/scratch/jobrunner/{jr_job_id}"
            remote_script = f"{remote_dir}/script.sbatch"

            # Handle both sbatch_args (new approach) and individual parameters (old approach)
            sbatch_args = []

            # If sbatch_args is provided, use it directly
            if 'sbatch_args' in data and data['sbatch_args']:
                sbatch_args = data['sbatch_args']
            else:
                # Build sbatch_args from individual parameters (old approach)
                if (partition := data.get('partition')) is not None:
                    sbatch_args.append(f"--partition={partition}")

                if (nodes := data.get('nodes')) is not None:
                    sbatch_args.append(f"--nodes={nodes}")

                if (ntasks := data.get('ntasks')) is not None:
                    sbatch_args.append(f"--ntasks={ntasks}")

                if (cpus_per_task := data.get('cpus_per_task')) is not None:
                    sbatch_args.append(f"--cpus-per-task={cpus_per_task}")

                if (mem := data.get('mem')) is not None:
                    sbatch_args.append(f"--mem={mem}")

                if (time_limit := data.get('time_limit')) is not None:
                    sbatch_args.append(f"--time={time_limit}")

                if (gpus_per_task := data.get('gpus_per_task')) is not None:
                    sbatch_args.append(f"--gpus-per-task={gpus_per_task}")

                if (ntasks_per_node := data.get('ntasks_per_node')) is not None:
                    sbatch_args.append(f"--ntasks-per-node={ntasks_per_node}")

                if data.get('exclusive'):
                    sbatch_args.append('--exclusive')

                if (export_val := data.get('export')) is not None:
                    sbatch_args.append(f"--export={export_val}")

                if (nodelist := data.get('nodelist')) is not None:
                    sbatch_args.append(f"-w {nodelist}")

                if (dependencies := data.get('dependency')) is not None:
                    dep = []
                    for event, job_id in dependencies:
                        dep.append(f"{event}:{job_id}")

                    dependency = ",".join(dep)
                    sbatch_args.append(f"--dependency={dependency}")

            # Add required arguments
            sbatch_args.extend([
                f"--job-name={job_name}",
                f"--comment=\"jr_job_id={jr_job_id}\"",
                f"--output={remote_dir.replace('~/', '')}/log.stdout",
                f"--error={remote_dir.replace('~/', '')}/log.stderr"
            ])

            # Create temporary file to hold the sbatch script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(script_content)
                script_path = f.name
                f.flush()

                # Create remote directory and copy script
                result = remote_command(SLURM_HOST, f"'mkdir -p {remote_dir}'")
                if not result['success']:
                    return jsonify({'error': f'Failed to copy script: {result["stderr"]}'}), 500

                scp_cmd = f"scp {script_path} mila:{remote_script}"
                subprocess.check_call(scp_cmd, shell=True)
            # ==

            sbatch_cmd = f"'sbatch {' '.join(sbatch_args)} -- {remote_script}'"

            # Create a temporary file to hold the sbatch command used
            # this is used for the re-execute this job
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(sbatch_cmd[1:-1])
                f.flush()
                cmd_path = f.name

                remote_cmd = f"{remote_dir}/cmd.sh"
                scp_cmd = f"scp {cmd_path} mila:{remote_cmd}"
                subprocess.check_call(scp_cmd, shell=True)

            # Submit job
            print(sbatch_cmd)
            result = remote_command(SLURM_HOST, sbatch_cmd)

            if result['success']:
                # Extract job ID from output
                job_id_match = re.search(r'Submitted batch job (\d+)', result['stdout'])
                job_id = job_id_match.group(1) if job_id_match else None

                return jsonify({
                    'success': True,
                    'job_id': job_id,
                    "jr_job_id": jr_job_id,
                    'message': result['stdout']
                })
            else:
                return jsonify({'error': result['stderr']}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Cancel job
    @app.route('/api/slurm/cancel/<job_id>', methods=['POST'])
    def api_slurm_cancel(job_id):
        """Cancel a Slurm job"""
        job_id = validate_slurm_job_id(job_id)
        try:
            result = remote_command(SLURM_HOST, f'scancel {job_id}')

            if result['success']:
                return jsonify({'success': True, 'message': f'Job {job_id} cancelled successfully'})
            else:
                return jsonify({'error': result['stderr']}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Synchronization mechanism for fetch_latest_job_acc
    _fetch_lock = threading.Lock()
    _last_fetch_time = 0
    _min_interval = 2.0  # 2 seconds minimum between executions

    def fetch_latest_job_acc(job_id):
        nonlocal _last_fetch_time

        with _fetch_lock:
            # Calculate time since last execution
            current_time = time.time()
            time_since_last = current_time - _last_fetch_time

            # If less than 2 seconds have passed, wait for the remaining time
            if time_since_last < _min_interval:
                sleep_time = _min_interval - time_since_last
                time.sleep(sleep_time)

            # Update the last execution time
            _last_fetch_time = time.time()

            print(f"Fetching job acc for {job_id}")
            # Try to get job logs using scontrol
            result = remote_command(SLURM_HOST, f'sacct --json -j {job_id}')

            if not result['success']:
                return {}

            jobs = json.loads(result['stdout'])["jobs"]

            if len(jobs) > 0:
                return jobs[0]

            return {}

    def fetch_latest_job_acc_cached(jr_job_id, job_id):
        data = fetch_latest_job_acc(job_id)

        if len(data) != 0:
            p = safe_job_path(jr_job_id, "meta", "acc.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fp:
                json.dump(data, fp)

        return data

    @app.route('/api/slurm/jobs/<jr_job_id>/acc/<job_id>')
    @local_cache("acc.json", "jr_job_id", check_validation=job_acc_cache_status)
    def api_slurm_job_acc(jr_job_id, job_id=None):
        """Get logs for a specific job"""
        try:
            if job_id is None:
                return ({})

            job_id = validate_slurm_job_id(job_id)
            return fetch_latest_job_acc(job_id)

        except Exception as e:
            return ({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/info/<job_id>')
    @local_cache("info.json", "jr_job_id")
    def api_slurm_job_info(jr_job_id, job_id=None):
        """Get logs for a specific job"""
        try:
            if job_id is None:
                return jsonify({})

            # Try to get job logs using scontrol
            job_id = validate_slurm_job_id(job_id)
            result = remote_command(SLURM_HOST, f'scontrol show job --json {job_id}')
            # result = remote_command(SLURM_HOST, f'sacct --json -j {job_id}')

            if not result['success']:
                return jsonify({'error': f'Failed to get job info: {result["stderr"]}'}), 500

            jobs = json.loads(result['stdout'])["jobs"]

            if len(jobs) > 0:
                return jsonify(jobs[0])

            return jsonify({})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def get_cached_state(jr_job_id, job_id):
        with cache_invalidator(jr_job_id, "acc.json", limit=30) as is_old:
            if not is_old:
                sacct_info = safe_job_path(jr_job_id, "meta", "acc.json")
                squeue_info = safe_job_path(jr_job_id, "meta", "info.json")

                if os.path.exists(sacct_info):
                    try:
                        with open(sacct_info, "r") as fp:
                            sacct = json.load(fp)
                            return get_acc_state(sacct)
                    except:
                        pass

                if os.path.exists(squeue_info):
                    try:
                        with open(squeue_info, "r") as fp:
                            squeue = json.load(fp)
                            return get_info_state(squeue)
                    except:
                        pass

        sacct = fetch_latest_job_acc_cached(jr_job_id, job_id)
        return get_acc_state(sacct)

    def is_job_state_terminal(jr_job_id, job_id):
        return is_state_terminal(get_cached_state(jr_job_id, job_id))

    @app.route('/api/slurm/jobs/<string:jr_job_id>/status/<int:job_id>')
    def api_slurm_job_status(jr_job_id, job_id):
        """Get the job Status"""

        cached_state = get_cached_state(jr_job_id, job_id)

        return {
            "status": cached_state
        }
  
    @app.route('/api/slurm/jobs/<jr_job_id>/info')
    @local_cache("info.json", "jr_job_id")
    def api_slurm_job_info_cached(jr_job_id):
        return api_slurm_job_info(jr_job_id=jr_job_id, job_id=None)

    @app.route('/api/slurm/jobs/<jr_job_id>/stdout/size')
    def api_slurm_job_stdout_size(jr_job_id):
        """Get logs for a specific job"""
        try:
            cache_dir = safe_job_path(jr_job_id)
            cache_file = os.path.join(cache_dir, "log.stdout")

            size = 0
            if os.path.exists(cache_file):
                size = os.path.getsize(cache_file)

            return jsonify({"size": size})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stdout')
    def api_slurm_job_stdout_base(jr_job_id):
        """Get logs for a specific job"""
        return api_slurm_job_stdout_extended(jr_job_id=jr_job_id)

    @app.route('/api/slurm/jobs/<jr_job_id>/stdout/<int:start>/<int:end>')
    @job_rsync_load("log.stdout", "jr_job_id")
    def api_slurm_job_stdout_extended(jr_job_id, start=None, end=None):
        """Get logs for a specific job"""
        try:
            return jsonify({
                "size": 0,
                "data": "",
                "status": "NA"
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr/size')
    def api_slurm_job_stderr_size(jr_job_id):
        """Get logs for a specific job"""
        try:
            cache_dir = safe_job_path(jr_job_id)
            cache_file = os.path.join(cache_dir, "log.stderr")
            size = 0
            if os.path.exists(cache_file):
                size = os.path.getsize(cache_file)

            return jsonify({"size": size})


        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr')
    @job_rsync_load("log.stderr", "jr_job_id")
    def api_slurm_job_stderr_base(jr_job_id):
        """Get logs for a specific job"""
        return api_slurm_job_stderr_extend(jr_job_id=jr_job_id)

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr/<int:start>/<int:end>')
    @job_rsync_load("log.stderr", "jr_job_id")
    def api_slurm_job_stderr_extend(jr_job_id, start=None, end=None):
        """Get logs for a specific job"""
        try:
            return jsonify({
                "size": 0,
                "data": "",
                "status": "NA"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stdout/tail')
    def api_slurm_job_stdout_tail(jr_job_id):
        """Get logs for a specific job"""
        try:
            log_path = f"scratch/jobrunner/{jr_job_id}/log.stdout"

            result = remote_command(SLURM_HOST, f'tail -n 100 {log_path}')

            return jsonify(result["stdout"])

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr/tail')
    def api_slurm_job_stderr_tail(jr_job_id):
        """Get logs for a specific job"""
        try:
            log_path = f"scratch/jobrunner/{jr_job_id}/log.stderr"

            result = remote_command(SLURM_HOST, f'tail -n 100 {log_path}')

            return jsonify(result["stdout"])

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Get available templates from slurm folder
    @app.route('/api/slurm/templates')
    def api_slurm_templates():
        """Get list of available templates from milabench/scripts/slurm folder"""
        try:
            templates = os.listdir(SLURM_TEMPLATES)

            return jsonify(templates)

        except Exception as e:
            return jsonify({'error': f'Failed to load templates: {str(e)}'}), 500

    # Get specific template content
    @app.route('/api/slurm/templates/<template_name>')
    def api_slurm_template_content(template_name):
        """Get content of a specific template"""
        try:
            script_file = safe_path(SLURM_TEMPLATES, template_name)

            if not os.path.exists(script_file):
                return jsonify({'error': f'Template {template_name} not found'}), 404

            with open(script_file, "r") as fp:
                content = fp.read()

            return jsonify({
                'name': template_name,
                'content': content
            })

        except Exception as e:
            return jsonify({'error': f'Failed to load template: {str(e)}'}), 500

    # Save custom template
    @app.route('/api/slurm/save-template', methods=['POST'])
    def api_slurm_save_template():
        """Save a custom template to the templates folder"""
        try:
            data = request.json
            template_name = data.get('name')
            content = data.get('content', '')

            if not template_name:
                return jsonify({'error': 'Template name is required'}), 400

            if not content:
                return jsonify({'error': 'Template content is required'}), 400

            # Create template file
            script_file = safe_path(SLURM_TEMPLATES, template_name)

            with open(script_file, "w") as fp:
                fp.write(content)

            return jsonify({
                'success': True,
                'message': f'Template {template_name} saved successfully'
            })

        except Exception as e:
            return jsonify({'error': f'Failed to save template: {str(e)}'}), 500

    # Get available Slurm config profiles
    @app.route('/api/slurm/profiles')
    def api_slurm_profiles():
        """Get available Slurm configuration profiles"""
        import yaml
        import os

        try:
            with open(SLURM_PROFILES, 'r') as f:
                configs = yaml.safe_load(f)

            profiles = []
            for profile_name, sbatch_args in configs.items():
                # Parse sbatch arguments into a more usable format
                parsed_config = {
                    'name': profile_name,
                    'sbatch_args': sbatch_args,
                    'parsed_args': _parse_sbatch_args(sbatch_args)
                }
                profiles.append(parsed_config)

            return jsonify(profiles)

        except Exception as e:
            return jsonify({'error': f'Failed to load profiles: {str(e)}'}), 500

    # Save custom profile
    @app.route('/api/slurm/save-profile', methods=['POST'])
    def api_slurm_save_profile():
        """Save a custom profile to the YAML file"""
        try:
            data = request.json
            profile_name = data.get('name')
            sbatch_args = data.get('sbatch_args', [])

            if not profile_name:
                return jsonify({'error': 'Profile name is required'}), 400

            # Filter out dependency-related arguments (safety check)
            # Dependencies are job-specific and should not be saved in reusable profiles
            filtered_args = [arg for arg in sbatch_args if not arg.startswith('--dependency')]

            # Read existing config
            with open(SLURM_PROFILES, 'r') as f:
                configs = yaml.safe_load(f)

            # Add new profile
            configs[profile_name] = filtered_args

            # Write back to file
            with open(SLURM_PROFILES, 'w') as f:
                yaml.dump(configs, f, default_flow_style=False, sort_keys=False)

            return jsonify({
                'success': True,
                'message': f'Profile {profile_name} saved successfully'
            })

        except Exception as e:
            return jsonify({'error': f'Failed to save profile: {str(e)}'}), 500


    @app.route('/api/slurm/pipeline/template/list')
    def api_pipeline_list():
        try:
            return jsonify([str(f[:-5]) for f in os.listdir(PIPELINE_DEF)])

        except Exception as e:
            return jsonify({'error': f'Failed to list pipeline: {str(e)}'}), 500

    @app.route('/api/slurm/pipeline/template/save', methods=['POST'])
    def api_pipeline_save():
        try:
            data = request.json

            pipeline_name = data.get("name")

            with open(safe_path(PIPELINE_DEF, pipeline_name) + ".json", "w") as fp:
                json.dump(data, fp, indent=2)

            return jsonify({"status": "ok"})
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Failed to save pipeline: {str(e)}'}), 500

    @app.route('/api/slurm/pipeline/template/load/<string:name>')
    def api_pipeline_load(name: str):
        try:
            return send_file(safe_path(PIPELINE_DEF, name + ".json"), mimetype="application/json")

        except Exception as e:
            return jsonify({'error': f'Failed to save pipeline: {str(e)}'}), 500

    def pipeline_definition_run(definition, context):
        # I propably want to be able insert some additional info
        # so jobs know their shared state and things like that

        pipeline = JobNode.from_json(definition)

        # reserve spot for all the jobs
        # Create job folders with their script
        # make the pipeline folder with links to the jobs
        #   copy the folers from local to remote
        #   rsync the pipeline folder as well
        pipeline.persist()

        # queue the jobs on slurm
        pipeline.schedule()

    @app.route('/api/slurm/pipeline/run/<string:name>', methods=['POST'])
    def api_pipeline_run_template(name):
        context = request.json["context"]

        with open(safe_path(PIPELINE_DEF, name + ".json"), "r") as fp:
            definition = json.load(fp)

        return pipeline_definition_run(definition, context)

    @app.route('/api/slurm/pipeline/run', methods=['POST'])
    def api_pipeline_run():
        context = request.json["context"]
        definition = request.json["definition"]

        return pipeline_definition_run(definition, context)
