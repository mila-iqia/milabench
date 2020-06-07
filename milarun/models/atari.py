import copy
import glob
import os
import time
from collections import deque
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule

from coleo import Argument, default
from milarun.lib import init_torch, coleo_main, dataloop, iteration_wrapper


@coleo_main
def main(exp):

    # Algorithm to use: a2c | ppo | acktr
    algorithm: Argument = default("a2c")

    # Gail epochs (default: 5)
    gail_epoch: Argument & int = default(5)

    # Learning rate (default: 7e-4)
    lr: Argument & float = default(7e-4)

    # Directory that contains expert demonstrations for gail
    gail_experts_dir: Argument = default("./gail_experts")

    # Gail batch size (default: 128)
    gail_batch_size: Argument & int = default(128)

    # Do imitation learning with gail
    gail: Argument & bool = default(False)

    # RMSprop optimizer epsilon (default: 1e-5)
    eps: Argument & float = default(1e-5)

    # RMSprop optimizer apha (default: 0.99)
    alpha: Argument & float = default(0.99)

    # discount factor for rewards (default: 0.99)
    gamma: Argument & float = default(0.99)

    # use generalized advantage estimation
    use_gae: Argument & bool = default(False)

    # gae lambda parameter (default: 0.95)
    gae_lambda: Argument & float = default(0.95)

    # entropy term coefficient (default: 0.01)
    entropy_coef: Argument & float = default(0.01)

    # value loss coefficient (default: 0.5)
    value_loss_coef: Argument & float = default(0.5)

    # max norm of gradients (default: 0.5)
    max_grad_norm: Argument & float = default(0.5)

    # sets flags for determinism when using CUDA (potentially slow!)
    cuda_deterministic: Argument & bool = default(False)

    # how many training CPU processes to use (default: 16)
    num_processes: Argument & int = default(16)

    # number of forward steps in A2C (default: 5)
    num_steps: Argument & int = default(5)

    # number of ppo epochs (default: 4)
    ppo_epoch: Argument & int = default(4)

    # number of batches for ppo (default: 32)
    num_mini_batch: Argument & int = default(32)

    # ppo clip parameter (default: 0.2)
    clip_param: Argument & float = default(0.2)

    # # log interval, one log per n updates (default: 10)
    # log_interval: Argument & int = default(10)

    # # save interval, one save per n updates (default: 100)
    # save_interval: Argument & int = default(100)

    # # eval interval, one eval per n updates (default: None)
    # eval_interval: Argument & int = default(None)

    # number of environment steps to train (default: 10e6)
    num_env_steps: Argument & int = default(10e6)

    # environment to train on (default: PongNoFrameskip-v4)
    env_name: Argument = default('PongNoFrameskip-v4')

    # directory to save agent logs (default: /tmp/gym)
    log_dir: Argument = default(None)

    # directory to save agent logs (default: ./trained_models/)
    save_dir: Argument = default('./trained_models/')

    # compute returns taking into account time limits
    use_proper_time_limits: Argument & bool = default(False)

    # use a recurrent policy
    recurrent_policy: Argument & bool = default(False)

    # use a linear schedule on the learning rate')
    use_linear_lr_decay: Argument & bool = default(False)

    # Seed to use
    seed: Argument & int = default(1234)

    # Number of iterations
    iterations: Argument & int = default(10)

    # we compute steps/sec
    batch_size = num_processes

    torch_settings = init_torch()
    device = torch_settings.device

    assert algorithm in ['a2c', 'ppo', 'acktr']

    if recurrent_policy:
        assert algorithm in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'


    num_updates = int(num_env_steps) // num_steps // num_processes

    envs = make_vec_envs(env_name, seed, num_processes,
                            gamma, log_dir, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                        base_kwargs={'recurrent': recurrent_policy})
    actor_critic.to(device)

    if algorithm == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, value_loss_coef,
                                entropy_coef, lr=lr,
                                eps=eps, alpha=alpha,
                                max_grad_norm=max_grad_norm)
    elif algorithm == 'ppo':
        agent = algo.PPO(actor_critic, clip_param, ppo_epoch, num_mini_batch,
                            value_loss_coef, entropy_coef, lr=lr,
                            eps=eps,
                            max_grad_norm=max_grad_norm)
    elif algorithm == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, value_loss_coef,
                                entropy_coef, acktr=True)

    rollouts = RolloutStorage(num_steps, num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(num_env_steps) // num_steps // num_processes

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    for it, j in dataloop(count(), wrapper=wrapper):
        it.set_count(batch_size)

        if use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr if algorithm == "acktr" else lr)

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                    for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0]for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()
        # ---
        rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        it.log(
            value_loss=value_loss,
            action_loss=action_loss,
        )

        rollouts.after_update()

        total_num_steps = (j + 1) * num_processes * num_steps

        # if j % log_interval == 0 and len(episode_rewards) > 1:
        #     end = time.time()
        #     print(
        #         "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
        #         format(j, total_num_steps,
        #             int(total_num_steps / (end - start)),
        #             len(episode_rewards),
        #             np.mean(episode_rewards),
        #             np.median(episode_rewards),
        #             np.min(episode_rewards),
        #             np.max(episode_rewards), dist_entropy,
        #             value_loss, action_loss))
    envs.close()
