import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SLURM_PROFILES = os.path.join(ROOT, 'config', 'clusters', 'slurm.yaml')
SLURM_TEMPLATES = os.path.join(ROOT, 'scripts', 'slurm')
PIPELINE_DEF = os.path.join(ROOT, 'scripts', 'pipeline')
JOBRUNNER_WORKDIR = "scratch/jobrunner"
JOBRUNNER_LOCAL_CACHE =  os.path.abspath(os.path.join(ROOT, '..', 'data'))

CLUSTERS = os.path.join(ROOT, "config", "clusters", "clusters.yml")