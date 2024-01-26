import json
import os
import argparse
from solver import Solver
from data_loader import get_loader
from synth import SyntheticData
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    # Data loader.
    celeba_loader = None
    rafd_loader = None
    synth_loader = None

    if config.dataset in ["CelebA", "Both"]:
        celeba_loader = get_loader(
            config.celeba_image_dir,
            config.attr_path,
            config.selected_attrs,
            config.celeba_crop_size,
            config.image_size,
            config.batch_size,
            "CelebA",
            config.mode,
            config.num_workers,
        )
    if config.dataset in ["RaFD", "Both"]:
        rafd_loader = get_loader(
            config.rafd_image_dir,
            None,
            None,
            config.rafd_crop_size,
            config.image_size,
            config.batch_size,
            "RaFD",
            config.mode,
            config.num_workers,
        )
    if config.dataset == "synth":

        def igen():
            return torch.rand((3, config.image_size, config.image_size)) * 2 - 1

        def ogen():
            return torch.randint(0, 2, (config.c_dim,)).to(torch.float)

        synth_dataset = SyntheticData(
            generators=[igen, ogen],
            n=config.batch_size,
            repeat=10000,
        )
        synth_loader = DataLoader(
            synth_dataset, batch_size=config.batch_size, num_workers=config.num_workers
        )

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, synth_loader, config)

    if config.mode == "train":
        if config.dataset in ["CelebA", "RaFD", "synth"]:
            solver.train()
        elif config.dataset in ["Both"]:
            solver.train_multi()
    elif config.mode == "test":
        if config.dataset in ["CelebA", "RaFD", "synth"]:
            solver.test()
        elif config.dataset in ["Both"]:
            solver.test_multi()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--c_dim", type=int, default=5, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument(
        "--c2_dim", type=int, default=8, help="dimension of domain labels (2nd dataset)"
    )
    parser.add_argument(
        "--celeba_crop_size",
        type=int,
        default=178,
        help="crop size for the CelebA dataset",
    )
    parser.add_argument(
        "--rafd_crop_size", type=int, default=256, help="crop size for the RaFD dataset"
    )
    parser.add_argument("--image_size", type=int, default=128, help="image resolution")
    parser.add_argument(
        "--g_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of G",
    )
    parser.add_argument(
        "--d_conv_dim",
        type=int,
        default=64,
        help="number of conv filters in the first layer of D",
    )
    parser.add_argument(
        "--g_repeat_num", type=int, default=6, help="number of residual blocks in G"
    )
    parser.add_argument(
        "--d_repeat_num", type=int, default=6, help="number of strided conv layers in D"
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=1,
        help="weight for domain classification loss",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=10, help="weight for reconstruction loss"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="weight for gradient penalty"
    )

    # Training configuration.
    parser.add_argument(
        "--dataset",
        type=str,
        default="synth",
        choices=["CelebA", "RaFD", "Both", "synth"],
    )
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=200000,
        help="number of total iterations for training D",
    )
    parser.add_argument(
        "--num_iters_decay",
        type=int,
        default=100000,
        help="number of iterations for decaying lr",
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0001, help="learning rate for G"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0001, help="learning rate for D"
    )
    parser.add_argument(
        "--n_critic", type=int, default=5, help="number of D updates per each G update"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--resume_iters", type=int, default=None, help="resume training from this step"
    )
    parser.add_argument(
        "--selected_attrs",
        "--list",
        nargs="+",
        help="selected attributes for the CelebA dataset",
        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
    )

    # Test configuration.
    parser.add_argument(
        "--test_iters", type=int, default=200000, help="test model from this step"
    )

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--use_tensorboard", type=str2bool, default=False)

    mbconfig = json.loads(os.environ["MILABENCH_CONFIG"])
    datadir = mbconfig["dirs"]["extra"]

    # Directories.
    parser.add_argument("--celeba_image_dir", type=str, default="data/celeba/images")
    parser.add_argument(
        "--attr_path", type=str, default="data/celeba/list_attr_celeba.txt"
    )
    parser.add_argument("--rafd_image_dir", type=str, default="data/RaFD/train")
    parser.add_argument("--log_dir", type=str, default=os.path.join(datadir, "logs"))
    parser.add_argument(
        "--model_save_dir", type=str, default=os.path.join(datadir, "models")
    )
    parser.add_argument(
        "--sample_dir", type=str, default=os.path.join(datadir, "samples")
    )
    parser.add_argument(
        "--result_dir", type=str, default=os.path.join(datadir, "results")
    )

    # Step size.
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=1000)
    parser.add_argument("--model_save_step", type=int, default=10000)
    parser.add_argument("--lr_update_step", type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
