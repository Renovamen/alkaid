import gym
import torch
from alkaid.agent import DQN
from alkaid.trainer import OffPolicyTrainer
from alkaid.net import QNet
from alkaid.env import GymWrapper

LR = 1e-4

env = GymWrapper(env=gym.make('LunarLander-v2'))

model = QNet(state_dim=env.state_dim, action_dim=env.action_dim)
optim = torch.optim.Adam(model.parameters(), lr=LR)
agent = DQN(model=model, optim=optim, env=env)

trainer = OffPolicyTrainer(env, agent, root="/Users/zou/Renovamen/Developing/alkaid/checkpoints/dqn")
trainer.train()
