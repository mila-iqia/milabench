def get_pytorch_version():
    def clean(k: str):
        pad = "  - "
        if k is None:
            return ""

        if k.startswith(pad):
            return k[len(pad) :].strip()
        return k.strip()

    def find_config(lines, key):
        for line in lines:
            if key in line:
                return clean(line)

        return None

    def parse_build_settings(settings):
        flags = dict()

        if settings is None:
            return flags

        _, settings = settings.split(":")
        for setting in settings.split(","):
            try:
                k, v = setting.split("=", maxsplit=1)
                flags[k.strip()] = v.strip()
            except ValueError:
                pass

        return flags

    try:
        import torch

        conf = torch.__config__.show().split("\n")

        compiler = conf[1]
        cpp = find_config(conf, "C++ Version")
        intel = find_config(conf, "oneAPI")
        mkl = find_config(conf, "OpenMP")
        openmp = find_config(conf, "OpenMP")
        lapack = find_config(conf, "LAPACK")
        nnpack = find_config(conf, "NNPACK")
        cpu = find_config(conf, "CPU")
        build_settings = find_config(conf, "Build settings")

        return dict(
            torch=torch.__version__,
            compiler=clean(compiler),
            cpp=clean(cpp),
            intel=clean(intel),
            mkl=clean(mkl),
            openmp=clean(openmp),
            lapack=clean(lapack),
            nnpack=clean(nnpack),
            cpu=clean(cpu),
            build_settings=parse_build_settings(build_settings),
        )

    except ImportError:
        return dict()


if __name__ == "__main__":
    import json

    print(json.dumps(get_pytorch_version()))
