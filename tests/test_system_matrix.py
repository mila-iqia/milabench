



from milabench.network import enable_offline
from milabench.system import multirun, build_system_config, option, apply_system, SizerOptions

from milabench.testing import official_config


def test_system_matrix():
    with enable_offline(True):
        sys = build_system_config(official_config("examples/system"))
        
        n = 0
        for name, conf in multirun():
            print(name, conf)
            n += 1

        assert n == 13


def test_apply_system_matrix():
    with enable_offline(True):
        sys = build_system_config(official_config("examples/system"))

        for name, conf in multirun():
            with apply_system(conf):
                print(conf)
                
                # Apply system worked and changed the config
                for k, v in conf.items():
                    assert option(k, lambda x: x) == v

                assert SizerOptions().save == option("sizer.save", lambda x: x)

    
    
if __name__ == "__main__":
    test_apply_system_matrix()
