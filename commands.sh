# ---
# Virtual Env
# ===========
export VIRTUAL_ENV="/Tmp/slurm.4123709.0/base/venv/torch"


# ---
# Milabench
# =========
export MILABENCH_DIR_BASE="/Tmp/slurm.4123709.0/base"
export MILABENCH_DIR_VENV="/Tmp/slurm.4123709.0/base/venv/torch"
export MILABENCH_DIR_DATA="/Tmp/slurm.4123709.0/base/data"
export MILABENCH_DIR_RUNS="/Tmp/slurm.4123709.0/base/runs"
export MILABENCH_DIR_EXTRA="/Tmp/slurm.4123709.0/base/extra/torchvision"
export MILABENCH_DIR_CACHE="/Tmp/slurm.4123709.0/base/cache"
export MILABENCH_CONFIG='{"system": {"arch": "cuda", "sshkey": null, "nodes": [{"ip": "127.0.0.1", "main": true, "name": "0", "port": 22, "user": "username", "hostname": "localhost", "aliaslist": [], "ipaddrlist": ["70:b5:e8:f0:5a:08", "fe80::1270:fd03:cd:a394%ibp161s0", "::1", "172.16.9.28", "fe80::72b5:e8ff:fef0:5a08%eno8303", "00:00:00:00:00:00", "00:00:02:5d:fe:80:00:00:00:00:00:00:10:70:fd:03:00:cd:a3:94", "10.20.9.28", "00:00:00:bf:fe:80:00:00:00:00:00:00:10:70:fd:03:00:e6:1b:38", "fe80::1270:fd03:e6:1b38%ibp37s0", "127.0.0.1", "10.20.137.28"], "local": true}], "gpu": {"capacity": "0 MiB"}, "self": {"ip": "127.0.0.1", "main": true, "name": "0", "port": 22, "user": "username", "hostname": "localhost", "aliaslist": [], "ipaddrlist": ["70:b5:e8:f0:5a:08", "fe80::1270:fd03:cd:a394%ibp161s0", "::1", "172.16.9.28", "fe80::72b5:e8ff:fef0:5a08%eno8303", "00:00:00:00:00:00", "00:00:02:5d:fe:80:00:00:00:00:00:00:10:70:fd:03:00:cd:a3:94", "10.20.9.28", "00:00:00:bf:fe:80:00:00:00:00:00:00:10:70:fd:03:00:e6:1b:38", "fe80::1270:fd03:e6:1b38%ibp37s0", "127.0.0.1", "10.20.137.28"], "local": true}}, "dirs": {"base": "/Tmp/slurm.4123709.0/base", "venv": "/Tmp/slurm.4123709.0/base/venv/torch", "data": "/Tmp/slurm.4123709.0/base/data", "runs": "/Tmp/slurm.4123709.0/base/runs", "extra": "/Tmp/slurm.4123709.0/base/extra/torchvision", "cache": "/Tmp/slurm.4123709.0/base/cache"}, "group": "torchvision", "install_group": "torch", "install_variant": "cuda", "run_name": "dev", "enabled": true, "capabilities": {"nodes": 1}, "max_duration": 600, "voir": {"options": {"stop": 60, "interval": "1s"}}, "validation": {"usage": {"gpu_load_threshold": 0.5, "gpu_mem_threshold": 0.5}}, "config_base": "/home/mila/d/delaunap/milabench/config", "config_file": "/home/mila/d/delaunap/milabench/config/standard.yaml", "definition": "/home/mila/d/delaunap/milabench/benchmarks/torchvision", "plan": {"method": "per_gpu"}, "argv": {"--precision": "tf32-fp16", "--lr": 0.01, "--no-stdout": true, "--epochs": 50, "--model": "resnet50", "--batch-size": 64}, "tags": ["classification", "convnet", "resnet", "vision"], "weight": 1.0, "name": "resnet50", "tag": ["resnet50"]}'

source $VIRTUAL_ENV/bin/activate

# ---
# resnet50
# ========
(
  python /home/mila/d/delaunap/milabench/benchmarks/torchvision/main.py --precision tf32-fp16 --lr 0.01 --no-stdout --epochs 10 --model resnet50 --batch-size 64
)

