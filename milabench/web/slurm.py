import os
import subprocess
import json
import re
from datetime import datetime
import tempfile
import os
import yaml
import uuid


from flask import request, jsonify, send_file

SLURM_PROFILES = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'slurm.yaml')

SLURM_TEMPLATES = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'slurm')


#   /home/$user/scratch/jobrunner/$internal_job_id/output.stdout
#   /home/$user/scratch/jobrunner/$internal_job_id/output.stderr
#   /home/$user/scratch/jobrunner/$internal_job_id/script.sbatch
#   /home/$user/scratch/jobrunner/$internal_job_id/output/...
#

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
            if key == 'partition':
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

        print(ssh_cmd)

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

    # Configuration for SSH connection to Slurm cluster
    SLURM_HOST = os.environ.get('SLURM_HOST', 'mila')


    @app.route('/api/slurm/jobs/persited')
    def api_slurm_persisted():
        """Get a list of job output still available"""
        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, "ls scratch/jobrunner")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 500

            jobs = result['stdout'].split(" ")
            
            return jsonify(jobs)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/all')
    def api_slurm_all_jobs():
        """Get list of all slurm jobs"""
        try:
            # Get running and pending jobs using JSON format
            result = remote_command(SLURM_HOST, "'sacct -u $USER --json'")

            if not result['success']:
                return jsonify({'error': f'Failed to get jobs: {result["stderr"]}'}), 500

            jobs = json.loads(result['stdout'])["jobs"]
            
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
                if 'comment' in job and job['comment']:
                    # Handle comment as string or object
                    comment_str = job['comment']
                    if isinstance(comment_str, dict):
                        # If comment is an object, try to get the string value
                        comment_str = comment_str.get('string', '') if isinstance(comment_str, dict) else str(comment_str)
                    elif not isinstance(comment_str, str):
                        comment_str = str(comment_str)

                    # Debug: print comment structure for first job
                    if job == jobs[0]:
                        print(f"DEBUG: Comment type: {type(job['comment'])}, value: {job['comment']}")
                        print(f"DEBUG: Processed comment_str: {comment_str}")

                    # Extract jr_job_id from comment like "jr_job_id=abc12345"
                    comment_match = re.search(r'jr_job_id=([a-zA-Z0-9]+)', comment_str)
                    if comment_match:
                        job['jr_job_id'] = comment_match.group(1)

            return jsonify(jobs)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Submit job
    @app.route('/api/slurm/submit', methods=['POST'])
    def api_slurm_submit():
        """Submit a new Slurm job"""
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Create a temporary SLURM script
            script_content = data.get('script', '')
            job_name = data.get('job_name', 'milabench_job')
            jr_job_id = str(uuid.uuid4())[:8]
            remote_dir = f"~/scratch/jobrunner/{jr_job_id}"
            remote_script = f"{remote_dir}/script.sbatch"

            # Handle both sbatch_args (new approach) and individual parameters (old approach)
            sbatch_args = []

            # If sbatch_args is provided, use it directly
            if 'sbatch_args' in data and data['sbatch_args']:
                sbatch_args = data['sbatch_args']
            else:
                # Build sbatch_args from individual parameters (old approach)
                if data.get('partition'):
                    sbatch_args.append(f"--partition={data['partition']}")
                if data.get('nodes'):
                    sbatch_args.append(f"--nodes={data['nodes']}")
                if data.get('ntasks'):
                    sbatch_args.append(f"--ntasks={data['ntasks']}")
                if data.get('cpus_per_task'):
                    sbatch_args.append(f"--cpus-per-task={data['cpus_per_task']}")
                if data.get('mem'):
                    sbatch_args.append(f"--mem={data['mem']}")
                if data.get('time_limit'):
                    sbatch_args.append(f"--time={data['time_limit']}")
                if data.get('gpus_per_task'):
                    sbatch_args.append(f"--gpus-per-task={data['gpus_per_task']}")
                if data.get('ntasks_per_node'):
                    sbatch_args.append(f"--ntasks-per-node={data['ntasks_per_node']}")
                if data.get('exclusive'):
                    sbatch_args.append('--exclusive')
                if data.get('export'):
                    sbatch_args.append(f"--export={data['export']}")
                if data.get('nodelist'):
                    sbatch_args.append(f"-w {data['nodelist']}")

            # Add required arguments
            sbatch_args.extend([
                f"--job-name={job_name}",
                f"--comment=\"jr_job_id={jr_job_id}\"",
                f"--output={remote_dir.replace('~/', '')}/log.stdout",
                f"--error={remote_dir.replace('~/', '')}/log.stderr"
            ])

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(script_content)
                script_path = f.name
                f.flush()

                # Create remote directory and copy script
                result = remote_command(SLURM_HOST, f"'mkdir -p {remote_dir}'")

                scp_cmd = f"scp {script_path} mila:{remote_script}"
                print(scp_cmd)
                subprocess.check_call(scp_cmd, shell=True)

                if not result['success']:
                    return jsonify({'error': f'Failed to copy script: {result["stderr"]}'}), 500

                # Submit job
                sbatch_cmd = f"'sbatch {' '.join(sbatch_args)} -- {remote_script}'"
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

    @app.route('/api/slurm/jobs/<job_id>/info')
    def api_slurm_job_info(job_id):
        """Get logs for a specific job"""
        try:
            # Try to get job logs using scontrol
            result = remote_command(SLURM_HOST, f'scontrol show job --json {job_id}')

            if not result['success']:
                return jsonify({'error': f'Failed to get job info: {result["stderr"]}'}), 500

            jobs = json.loads(result['stdout'])["jobs"]

            if len(jobs) > 0:
                return jsonify(jobs[0])

            return jsonify({})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<job_id>/stdout')
    def api_slurm_job_stdout(job_id):
        """Get logs for a specific job"""
        try:
            log_path = f"scratch/jobrunner/{job_id}/log.stdout"

            result = remote_command(SLURM_HOST, f'tail -n 100 {log_path}')

            return jsonify(result["stdout"])

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/slurm/jobs/<job_id>/stderr')
    def api_slurm_job_stderr(job_id):
        """Get logs for a specific job"""
        try:
            log_path = f"scratch/jobrunner/{job_id}/log.stderr"

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

            # Read existing config
            with open(SLURM_PROFILES, 'r') as f:
                configs = yaml.safe_load(f)

            # Add new profile
            configs[profile_name] = sbatch_args

            # Write back to file
            with open(SLURM_PROFILES, 'w') as f:
                yaml.dump(configs, f, default_flow_style=False, sort_keys=False)

            return jsonify({
                'success': True,
                'message': f'Profile {profile_name} saved successfully'
            })

        except Exception as e:
            return jsonify({'error': f'Failed to save profile: {str(e)}'}), 500
