# -*- coding: utf-8 -*-
"""Orion + Brax Training with PyTorch on GPU

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KlaTeScstmRg7AIWLgrXy9zGmayb5zMS
"""
import argparse
import os

from giving import give, given

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import torch  # This is a bit of a trick to make jax use torch's packaged libs
from brax import envs
from brax.training.agents.ppo.train import train


def run():
    parser = argparse.ArgumentParser(description="Brax training")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ant",
        help="Environment to simulate",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=100_000_000,
    )
    parser.add_argument(
        "--discounting",
        type=float,
        default=0.97,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )
    parser.add_argument(
        "--entropy-cost",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--unroll-length",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    train(
        environment=envs.get_environment(env_name=args.env),
        num_timesteps=args.num_timesteps,
        discounting=args.discounting,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        normalize_observations=True,
        action_repeat=1,
        progress_fn=lambda n, metrics: give(**metrics),
        num_evals=args.num_evals,
        reward_scaling=args.reward_scaling,
        episode_length=args.episode_length,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
    )


def main():
    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    with given() as gv:
        gv["?training/sps"].map(
            lambda sps: {"task": "train", "rate": sps, "units": "steps/s"}
        ).give()
        gv["?eval/episode_reward"].map(lambda reward: -reward.item()).as_("loss").give()
        main()
