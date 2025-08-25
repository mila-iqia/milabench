from __future__ import annotations

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

from cantilever.core.timer import timeit
from flask import request, jsonify, send_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SLURM_PROFILES = os.path.join(ROOT, 'config', 'slurm.yaml')
SLURM_TEMPLATES = os.path.join(ROOT, 'scripts', 'slurm')
JOBRUNNER_WORKDIR = "scratch/jobrunner"
JOBRUNNER_LOCAL_CACHE =  os.path.abspath(os.path.join(ROOT, '..', 'data'))

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


class Pipeline:
    """The job runner define a dependency between jobs and schedule them to slurm.
    It is able to launch a job on the cluster.

    Its most interesting feature is being able to track which job failed and which were successful 
    and only trigger the failed one.

    To do so it simply iterate over the job tree and mark the one failed to retry and skip the successful ones.
    """

    def __init__(self, job_definition):
        self.definition: JobNode = job_definition

        # A pipeline run is something only known to the job runner and as such does not have Slurm Job ID
        # But it has a jobs id
        self.job_id = None

    def schedule(self):
        pass

    def rerun(self) -> Pipeline:
        # traverse the job definition and create a new one
        pass

    def __json__(self):
        return {
            "type": "pipeline",
            "definition": self.definition.__json__(),
            "job_id": self.job_id
        }
 

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
                return Pipeline(JobNode.from_json(data.pop("definition")), **data)
            case unknown_type:
                raise RuntimeError(f"Unknown type: {unknown_type}")


class Job(JobNode):
    def __init__(self, script, profile, job_id=None, slurm_jobid=None):
        self.script = script
        self.profile = profile
        self.job_id = job_id
        self.slurm_jobid = slurm_jobid

    def gen(self, depends_event=AFTER_OK, depends_on=None):
        args = []

        if depends_on is not None:
            args.append(f"--dependency={depends_event}:{depends_on}")

        # Here we submit the job definition to slurm itslef get the jobid from the query
        # also make a job runner id for it
        cmd = "squeue ..."
    
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
    def __init__(self, *jobs):
        self.jobs = jobs

    def gen(self, depends_on=None):
        job_id = depends_on

        for job in self.jobs:
            job_id = job.gen(depends_on=job_id)

        return job_id

    def __json__(self):
        return {
            "type": "sequential",
            "jobs": [
                job.__json__() for job in self.jobs
            ]
        }


class Parallel(JobNode):
    def __init__(self, *jobs):
        self.jobs = jobs
    
    def gen(self, depends_on=None):
        job_ids = []
        for job in self.jobs: 
            job_ids.append(job.gen(depends_on=depends_on))

        return ':'.join(job_ids)

    def __json__(self):
        return {
            "type": "parallel",
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
    


def job_acc_cache_status(filename: str):
    with open(filename, "r") as fp:
        try:
            job_acc = json.load(fp)

            job_states = job_acc.get("state", {}).get("current", [])

            # Cache is good
            if "COMPLETED" in job_states:
                return True
            
            return False
        except json.decoder.JSONDecodeError:
            return False


def local_cache(cache_key, arg_key="job_id", check_validation=None):
    """Cache a remote result locally"""
    def wrapped(fun):
        @wraps(fun)
        def wrapper(**kwargs):
            job_id = kwargs.get(arg_key)

            cache_dir = os.path.join(JOBRUNNER_LOCAL_CACHE, job_id, "meta")
            cache_file = os.path.join(cache_dir, cache_key)

            try:
                if os.path.exists(cache_file):
                    if check_validation is None or check_validation(cache_file):
                        return send_file(cache_file, mimetype="application/json")
            except:
                traceback.print_exc()

            result = fun(**kwargs)

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


def rsync_jobrunner_folder():
    try:
        rsync_cmd = f"rsync -az mila:{JOBRUNNER_WORKDIR}/ {JOBRUNNER_LOCAL_CACHE}"

        with timeit("rsync") as chrono:
            _ = subprocess.run(
                    rsync_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=True
                )
            
        print(f"{rsync_cmd} {chrono.timing.value.avg} s")

        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


def rsync_load(cache_key, arg_key="job_id"):
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

            cache_dir = os.path.join(JOBRUNNER_LOCAL_CACHE, job_id)
            cache_file = os.path.join(cache_dir, cache_key)

            if rsync_jobrunner_folder() == 0:
                if os.path.exists(cache_file):
                    with open(cache_file, "r") as fp:
                        if start is not None and end is not None:
                            fp.seek(start, os.SEEK_SET)
                            return {
                                "data": fp.read(end - start),
                                "size": os.path.getsize(cache_file)   
                            }

                        return {
                            "data": fp.read(),
                            "size": os.path.getsize(cache_file)    
                        }

            return ""
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


def remote_command(host, command):
    """Execute a Slurm command via SSH"""
    try:
        # Remote execution via SSH
        # ssh_cmd = ['ssh', SLURM_HOST, command]
        ssh_cmd = f'ssh {host} {command}'

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        import traceback
        traceback.print_exc()
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


def slurm_integration(app):
    """Add Slurm integration routes to the Flask app"""

    app.scheduler.add_job(rsync_jobrunner_folder, 'interval', seconds=60)

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

            return jsonify(jobs)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/persited')
    def api_slurm_persisted():
        return api_slurm_persisted_limited(100)

    @app.route('/api/slurm/jobs/persited/<int:limit>')
    def api_slurm_persisted_limited(limit=None):
        """Get a list of job output still available"""
        try:
            if rsync_jobrunner_folder() == 0:
                # jobs = os.listdir(JOBRUNNER_LOCAL_CACHE)
                # jobs.remove(".git")
                result = subprocess.check_output(f"ls -t {JOBRUNNER_LOCAL_CACHE}", shell=True, text=True)
                jobs = list(filter(lambda x: x != "", result.split("\n")))

                if limit is None:
                    return jobs

                return jobs[:limit]
            return []

        except Exception as e:
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
    def api_slurm_job_status(job_id):
        """Get list of all slurm jobs"""
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
            if rsync_jobrunner_folder() == 0:
                cache_dir = os.path.join(JOBRUNNER_LOCAL_CACHE, jr_job_id)

                cmd = f"git add {cache_dir} && git commit -m \"{message}\" && git push origin main"

                subprocess.check_call(cmd, shell=True, cwd=cache_dir)

            return jsonify({})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/rerun/<string:jr_job_id>')
    def api_slurm_rerun(jr_job_id: str):
        """Rerun a previous job"""

        old_jr_job_id = jr_job_id
        new_jr_job_id = generate_unique_job_name(from_job_id=jr_job_id)

        old_remote_dir = f"~/scratch/jobrunner/{old_jr_job_id}"
        new_remote_dir = f"~/scratch/jobrunner/{new_jr_job_id}"

        old_remote_script = f"{old_remote_dir}/script.sbatch"
        new_remote_script = f"{new_remote_dir}/script.sbatch"

        old_cmd_path = f"{old_remote_dir}/cmd.sh"

        # Create remote directory and copy script
        result = remote_command(SLURM_HOST, f"'mkdir -p {new_remote_dir}'")
        if not result['success']:
            return jsonify({'error': f'Failed to create job directory: {result["stderr"]}'}), 500

        # Copy old_remote_script to new job
        result = remote_command(SLURM_HOST, f"'cp {old_remote_script} {new_remote_script}'")
        if not result['success']:
            return jsonify({'error': f'Failed to copy script: {result["stderr"]}'}), 500

        result = remote_command(SLURM_HOST, f"'cat {old_cmd_path}'")
        if not result['success']:
            return jsonify({'error': f'Failed to retrieve previous job command: {result["stderr"]}'}), 500

        old_cmd = result["stdout"]

        new_cmd = "'" + old_cmd.replace(old_jr_job_id, new_jr_job_id) + "'"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(new_cmd[1:-1])
            f.flush()
            cmd_path = f.name

            remote_cmd = f"{new_remote_dir}/cmd.sh"
            scp_cmd = f"scp {cmd_path} mila:{remote_cmd}"
            subprocess.check_call(scp_cmd, shell=True)

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
        try:
            result = remote_command(SLURM_HOST, f'scancel {job_id}')

            if result['success']:
                return jsonify({'success': True, 'message': f'Job {job_id} cancelled successfully'})
            else:
                return jsonify({'error': result['stderr']}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/acc/<job_id>')
    @local_cache("acc.json", "jr_job_id", check_validation=job_acc_cache_status)
    def api_slurm_job_acc(jr_job_id, job_id=None):
        """Get logs for a specific job"""
        try:
            if job_id is None:
                return jsonify({})

            # Try to get job logs using scontrol
            result = remote_command(SLURM_HOST, f'sacct --json -j {job_id}')

            if not result['success']:
                return jsonify({'error': f'Failed to get job info: {result["stderr"]}'}), 500

            jobs = json.loads(result['stdout'])["jobs"]

            if len(jobs) > 0:
                return jsonify(jobs[0])

            return jsonify({})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/info/<job_id>')
    @local_cache("info.json", "jr_job_id")
    def api_slurm_job_info(jr_job_id, job_id=None):
        """Get logs for a specific job"""
        try:
            if job_id is None:
                return jsonify({})

            # Try to get job logs using scontrol
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

    @app.route('/api/slurm/jobs/<jr_job_id>/info')
    @local_cache("info.json", "jr_job_id")
    def api_slurm_job_info_cached(jr_job_id):
        return api_slurm_job_info(jr_job_id, None)

    @app.route('/api/slurm/jobs/<jr_job_id>/stdout/size')
    def api_slurm_job_stdout_size(jr_job_id):
        """Get logs for a specific job"""
        try:
            cache_dir = os.path.join(JOBRUNNER_LOCAL_CACHE, jr_job_id)
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
    @rsync_load("log.stdout", "jr_job_id")
    def api_slurm_job_stdout_extended(jr_job_id, start=None, end=None):
        """Get logs for a specific job"""
        try:
            return jsonify("")

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr/size')
    def api_slurm_job_stderr_size(jr_job_id):
        """Get logs for a specific job"""
        try:
            cache_dir = os.path.join(JOBRUNNER_LOCAL_CACHE, jr_job_id)
            cache_file = os.path.join(cache_dir, "log.stderr")
            size = 0
            if os.path.exists(cache_file):
                size = os.path.getsize(cache_file)

            return jsonify({"size": size})


        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr')
    @rsync_load("log.stderr", "jr_job_id")
    def api_slurm_job_stderr_base(jr_job_id):
        """Get logs for a specific job"""
        return api_slurm_job_stderr_extend(jr_job_id=jr_job_id)

    @app.route('/api/slurm/jobs/<jr_job_id>/stderr/<int:start>/<int:end>')
    @rsync_load("log.stderr", "jr_job_id")
    def api_slurm_job_stderr_extend(jr_job_id, start=None, end=None):
        """Get logs for a specific job"""
        try:
            return jsonify("")
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
            script_file = os.path.join(SLURM_TEMPLATES, template_name)

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
            script_file = os.path.join(SLURM_TEMPLATES, template_name)

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
