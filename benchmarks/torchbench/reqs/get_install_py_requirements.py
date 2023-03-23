# Fake setup.py to be used with piptools compile
# This file is used in reqs/pip-compile.sh and offload some of the logic for fix
# the install of some requirements
import os

_TORCHBENCHMARK_PATH = os.environ.get("_MILABENCH_TBPATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "torchbenchmark"))
_MODEL=os.path.basename(os.path.dirname(__file__))
_MODEL_PATH=os.path.join(_TORCHBENCHMARK_PATH, "models", str(_MODEL))


if _MODEL == "pytorch_unet":
    with open(os.path.join(_MODEL_PATH, "pytorch_unet/requirements.txt")) as f:
        require_packages = [line[:-1] if line[-1] == "\n" else line for line in f]
elif os.path.exists(os.path.join(_MODEL_PATH, "requirements.txt")):
    with open(os.path.join(_MODEL_PATH, "requirements.txt")) as f:
        require_packages = [line[:-1] if line[-1] == "\n" else line for line in f]
else:
    require_packages = []

for i in range(len(require_packages)):
    if require_packages[i] == "git+https://github.com/facebookresearch/detectron2.git@c470ca3":
        # `detectron2` has a setup.py which requires `torch` to be installed.
        # This is apparently incompatible with piptools compile which creates a
        # clean venv in which `torch` is not present
        # Replace that dependance with the content of `install_requires`
        import sys
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests'], check=True, stdout=subprocess.DEVNULL)
        import requests
        detectron2_requires = []
        r = requests.get("https://raw.githubusercontent.com/facebookresearch/detectron2/c470ca3/setup.py")
        for l in r.text.split("\n"):
            if l.strip(" ").startswith("extras_require"):
                break
            elif detectron2_requires or l.strip(" ").startswith("install_requires"):
                req = l.strip(" ").split("#")[0].split(",")[0].strip("\"")
                if req:
                    detectron2_requires.append(req)
        # setup.py programmatically checks for the torch version
        require_packages = require_packages[:i] + ["torch>=1.8"] + detectron2_requires[1:-1] + require_packages[i+1:]
        break

for r in require_packages:
    print(r)