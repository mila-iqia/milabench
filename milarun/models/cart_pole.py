import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from coleo import Argument, default
from milarun.lib import init_torch, coleo_main, dataloop, iteration_wrapper


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


@coleo_main
def main(exp):
    # discount factor (default: 0.99)
    gamma: Argument & float = default(0.99)

    # render the environment
    render: Argument & bool = default(False)

    # seed for the environment
    seed: Argument & int = default(1234)

    # length of one episode
    episode_length: Argument & int = default(500)

    torch_settings = init_torch()
    device = torch_settings.device

    env = gym.make('CartPole-v0')
    env.seed(seed)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    print(torch_settings)

    def select_action(state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item()


    def finish_episode():
        R = 0
        policy_loss = []
        returns = []

        for r in policy.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        del policy.rewards[:]
        del policy.saved_log_probs[:]


    running_reward = 10

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    for it, _ in dataloop(count(), wrapper=wrapper):
        it.set_count(episode_length)

        state, ep_reward = env.reset(), 0

        for t in range(episode_length):

            action = select_action(state)

            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward

            # we actually do not care about solving the thing
            if done:
                state, ep_reward = env.reset(), 0

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        it.log(reward=running_reward)
        finish_episode()
