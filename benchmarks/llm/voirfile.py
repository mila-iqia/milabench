from dataclasses import dataclass

from voir.phase import StopProgram
from voir import configurable
from benchmate.observer import BenchObserver
from benchmate.monitor import voirfile_monitor
from benchmate.benchrun import forward_voir_file

@dataclass
class Config1:
    """voir configuration"""

    # Whether to display the dash or not
    dash: bool = False

    # How often to log the rates
    interval: str = "1s"

    # Number of rates to skip before logging
    skip: int = 5

    # Number of rates to log before stopping
    stop: int = 20

    # Number of seconds between each gpu poll
    gpu_poll: int = 3


def lora_single_device(ov, observer):
    try:
        def wrap_dataloader(args):
            sampler, loader = args
            wrapped = observer.loader(loader, custom_step=True)
            return sampler, wrapped
        
        def wrap_lr_scheduler(scheduler):
            original = scheduler.step

            def newstep(*args, **kwargs):
                original(*args, **kwargs)
                # observer.step()

            scheduler.step = newstep
            return scheduler
        
        def wrap_loss(loss):
            observer.record_loss(loss)
            observer.step()
            return loss

        probe = ov.probe("//LoRAFinetuneRecipeSingleDevice/_setup_data() as loader", overridable=True)
        probe['loader'].override(wrap_dataloader)

        probe = ov.probe("//LoRAFinetuneRecipeSingleDevice/_setup_lr_scheduler() as scheduler", overridable=True)
        probe['scheduler'].override(wrap_lr_scheduler)

        probe = ov.probe("//LoRAFinetuneRecipeSingleDevice/train > loss_to_log", overridable=True)
        probe['loss_to_log'].override(wrap_loss)
    except:
        pass



def lora_distributed(ov, observer):
    def wrap_dataloader(args):
        sampler, loader = args
        wrapped = observer.loader(loader, custom_step=True)
        return sampler, wrapped
    
    def wrap_lr_scheduler(scheduler):
        original = scheduler.step

        def newstep(*args, **kwargs):
            original(*args, **kwargs)
            # observer.step()

        scheduler.step = newstep
        return scheduler
    
    def wrap_loss(loss):
        observer.record_loss(loss)
        observer.step()
        return loss

    probe = ov.probe("//LoRAFinetuneRecipeDistributed/_setup_data() as loader", overridable=True)
    probe['loader'].override(wrap_dataloader)

    probe = ov.probe("//LoRAFinetuneRecipeDistributed/_setup_lr_scheduler() as scheduler", overridable=True)
    probe['scheduler'].override(wrap_lr_scheduler)

    probe = ov.probe("//LoRAFinetuneRecipeDistributed/train > loss_to_log", overridable=True)
    probe['loss_to_log'].override(wrap_loss)


@configurable
def instrument_main(ov, options: Config1):
    yield ov.phases.init

    yield ov.phases.load_script

    with forward_voir_file():
        try:
            yield ov.phases.run_script
        except StopProgram:
            print("early stopped")