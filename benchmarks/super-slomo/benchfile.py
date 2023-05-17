from milabench.pack import Package


class SuperSlomoPack(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "slomo/train.py"


__pack__ = SuperSlomoPack
