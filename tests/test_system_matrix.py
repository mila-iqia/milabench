




from milabench.system import multirun, build_system_config, enable_offline, option, apply_system, SizerOptions

from milabench.testing import official_config


def test_system_matrix():
    with enable_offline(True):
        sys = build_system_config(official_config("examples/system"))
        
        n = 0
        for name, conf in multirun():
            print(name, conf)
            n += 1

        assert n == 39


def test_apply_system_matrix():
    with enable_offline(True):
        sys = build_system_config(official_config("examples/system"))

        for name, conf in multirun():
            with apply_system(conf):
                
                # Apply system worked and changed the config
                for k, v in conf.items():
                    assert option(k, lambda x: x) == v

    
    
if __name__ == "__main__":
    test_apply_system_matrix()
