
from omegaconf import DictConfig
import torchcompat.core as acc
from torchtune import config, training


def prepare_voir(recipe):
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor

    def batch_size(x):
        bs, token = x["tokens"].shape
        return bs * token

    observer = BenchObserver(
        earlystop=30,
        raise_stop_program=True,
        batch_size_fn=batch_size,
        stdout=True
    )

    def on_loss(loss):
        observer.record_loss(loss)
        observer.step()

    recipe._dataloader = observer.loader(recipe._dataloader, custom_step=True)
    recipe.log_loss = on_loss

    return observer, bench_monitor


def recipe_distributed_main(recipe_name, recipe_cls, cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    
    # init_process_group("cuda:nccl,cpu:gloo")
    acc.init_process_group()

    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name=recipe_name, cfg=cfg)

    recipe = recipe_cls(cfg=cfg)
    recipe.setup(cfg=cfg)

    from voir.phase import StopProgram

    try:
        _, monitor = prepare_voir(recipe)
        with monitor():
            recipe.train()
    
    except StopProgram:
        print("early stopping")

    recipe.cleanup()



# @config.parse
# def recipe_main(cfg: DictConfig) -> None:
#     """
#     Entry point for the recipe.

#     Configurable parameters are read in the following order:
#         - Parameters specified in config (see available configs through ``tune ls``)
#         - Overwritten by arguments from the command-line
#     """
#     import sys

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(os.path.join(current_dir))

#     from .utils import recipe_distributed_main

#     return recipe_distributed_main(
#         "FullFinetuneRecipeDistributed",
#         FullFinetuneRecipeDistributed,
#         cfg
#     )



# if __name__ == "__main__":
#     sys.exit(recipe_main())
