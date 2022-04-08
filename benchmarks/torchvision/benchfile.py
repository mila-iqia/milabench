from milabench.pack import Package


class TheBenchmark(Package):
    requirements_file = "requirements.txt"
    prepare_script = "prepare.py"
    main_script = "main.py"


__pack__ = TheBenchmark
