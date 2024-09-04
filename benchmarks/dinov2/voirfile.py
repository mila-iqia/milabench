from dataclasses import dataclass

from voir.phase import StopProgram
from voir import configurable
from benchmate.observer import BenchObserver
from benchmate.monitor import voirfile_monitor


@dataclass
class Config:
    """voir configuration"""

    # Whether to display the dash or not
    dash: bool = False

    # How often to log the rates
    interval: str = "1s"

    # Number of rates to skip before logging
    skip: int = 5

    # Number of rates to log before stopping
    stop: int = 60

    # Number of seconds between each gpu poll
    gpu_poll: int = 3


def populate_slurm():
    import json
    import os

    config = json.loads(os.environ["MILABENCH_CONFIG"])

    nodes = [n["name"] for n in config["system"]["nodes"]]

    env = {
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_NODELIST": ",".join(nodes),
        "SLURM_NTASKS": str(len(config["system"]["nodes"])),
        "SLURM_PROCID":  "2",   # RANK
        "SLURM_LOCALID": "1",   # Local RANK
    }


@configurable
def instrument_main(ov, options: Config):
    import os

    yield ov.phases.init

    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")

    yield ov.phases.load_script

    # GPU monitor, rate, loss etc...
    voirfile_monitor(ov, options)

    code_patch(ov)

    #
    # Insert milabench tools
    #
    def batch_size(x):
        return x["collated_global_crops"].shape[0]

    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=batch_size,
    )

    probe = ov.probe("/dinov2.data.loaders/make_data_loader() as loader", overridable=True)
    probe['loader'].override(observer.loader)

    probe = ov.probe("/dinov2.train.train/do_train > losses_reduced", overridable=True)
    probe["losses_reduced"].override(observer.record_loss)

    probe = ov.probe("/dinov2.train.train/build_optimizer() as optimizer", overridable=True)
    probe['optimizer'].override(observer.optimizer)
    
    #
    # Run the benchmark
    #
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")



def code_patch(ov):
    # FIX dinov2 code using ptera
    import os
    
    from torchvision.datasets import ImageFolder
    import torch
    import dinov2.train.train 

    class SSLMetaArch2(dinov2.train.train.SSLMetaArch):
        def fsdp_synchronize_streams(self):
            if self.need_to_synchronize_fsdp_streams:
                torch.cuda.synchronize()
                self.need_to_synchronize_fsdp_streams = False


    dinov2.train.train.SSLMetaArch = SSLMetaArch2
    dinov2.train.ssl_meta_arch.reshard_fsdp_model = lambda *args: None

    probe = ov.probe("/dinov2.distributed/_is_slurm_job_process() as is_slrum", overridable=True)
    probe['is_slrum'].override(lambda *args: False)

    def override_parsed_dataset(results):
        class_, kwargs = results
        return ImageFolder, {"root": os.path.join(kwargs["root"], "train")}

    probe = ov.probe("/dinov2.data.loaders/_parse_dataset_str() as dataset_kwargs", overridable=True)
    probe['dataset_kwargs'].override(override_parsed_dataset)
