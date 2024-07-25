from dataclasses import dataclass

from voir.phase import StopProgram
from voir import configurable
from voir.instruments import dash, early_stop, log
from benchmate.monitor import monitor_monogpu
from benchmate.observer import BenchObserver


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


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    import os
    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")

    yield ov.phases.load_script

    if options.dash:
        ov.require(dash)

    if int(os.getenv("RANK", 0)) == 0:
        ov.require(
            log("value", "progress", "rate", "units", "loss", "gpudata", context="task"),
            early_stop(n=options.stop, key="rate", task="train"),
            monitor_monogpu(poll_interval=options.gpu_poll),
        )

    #
    # Insert milabench tools
    #
    def batch_size(x):
        return x["collated_global_crops"].shape[0]

    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=batch_size,
    )

    # Prevent dinov2 from recognizing slurm 
    probe = ov.probe("/dinov2.distributed/_is_slurm_job_process() as is_slrum", overridable=True)
    probe['is_slrum'].override(lambda *args: False)


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
        print(results)
        class_, kwargs = results
        return ImageFolder, {"root": os.path.join(kwargs["root"], "train")}

    probe = ov.probe("/dinov2.data.loaders/_parse_dataset_str() as dataset_kwargs", overridable=True)
    probe['dataset_kwargs'].override(override_parsed_dataset)

    probe = ov.probe("/dinov2.data.loaders/make_data_loader() as loader", overridable=True)
    probe['loader'].override(observer.loader)

    # probe = ov.probe("//my_criterion_creator() as criterion", overridable=True)
    # probe['criterion'].override(observer.criterion)

    # probe = ov.probe("//my_optimizer_creator() as optimizer", overridable=True)
    # probe['optimizer'].override(observer.optimizer)
    
    #
    # Run the benchmark
    #
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")