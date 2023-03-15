from milabench import gpu
from milabench.pack import Package

# This is called to ensure the arch variable is set
gpu.get_gpu_info()


class PLBenchmark(Package):
    # Requirements file installed by install(). It can be empty or absent.
    if gpu.arch == "cuda":
        requirements_file = "requirements.cuda.txt"
    elif gpu.arch == "rocm":
        requirements_file = "requirements.amd.txt"
    else:
        raise ValueError(f"Unsupported arch: {gpu.arch}")

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = "main.py"


__pack__ = PLBenchmark
