import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import gym
import torch
from alkaid.agent import DQN, DuelingDQN, DoubleDQN
from alkaid.trainer import OffPolicyTrainer
from alkaid.net import QNet
from alkaid.env import GymWrapper
from alkaid.utils import Logger, Ploter

AGENT_LIST = {
    'dqn': DQN,
    'dueling-dqn': DuelingDQN,
    'double-dqn': DoubleDQN
}

LR = 1e-4
AGENT = 'dqn'

if __name__ == '__main__':
    env = GymWrapper(env=gym.make('LunarLander-v2'))
    model = QNet(state_dim=env.state_dim, action_dim=env.action_dim)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    agent = AGENT_LIST[AGENT](model=model, optim=optim, env=env)
    logger = Logger(root=os.path.join(base_path, "logs/dqn"))
    ploter = Ploter(root=os.path.join(base_path, "figures/dqn"))

    trainer = OffPolicyTrainer(
        agent, env,
        root = os.path.join(base_path, "checkpoints/dqn"),
        logger = logger,
        ploter = ploter
    )
    trainer.train()
