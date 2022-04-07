from milabench.pack import Package


class PLBenchmark(Package):
    # Requirements file installed by install(). It can be empty or absent.
    requirements_file = "frozen_reqs.txt"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = "main.py"


__pack__ = PLBenchmark
