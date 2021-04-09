import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import gym
import torch
from alkaid.agent import DQN
from alkaid.trainer import OffPolicyTrainer
from alkaid.net import QNet
from alkaid.env import GymWrapper

LR = 1e-4

if __name__ == '__main__':
    env = GymWrapper(env=gym.make('LunarLander-v2'))
    model = QNet(state_dim=env.state_dim, action_dim=env.action_dim)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    agent = DQN(model=model, optim=optim, env=env)

    trainer = OffPolicyTrainer(env, agent, root=os.path.join(base_path, "checkpoints/dqn"))
    trainer.train()
