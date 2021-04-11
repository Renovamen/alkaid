import os
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Optional

import gym
import numpy as np
import torch

from ..utils import get_datetime, mkdir

class Trainer(ABC):
    """
    Base class for all trainers.

    Args:
        agent: Agent
        env (gym.Env): Environment
        root (str): Directory where the checkpoints and logs should be saved
        max_timesteps (int, optional, default=2**20): Maximum limit of
            timesteps to train for
        stop_if_reach_goal (bool, optional, default=True): Stop when reach
            ``env.target_reward`` or not
        eval_episodes (int, optional, default=4): Number of episodes to evaluate for
        print_gap (int, optional, default=2**11): Timesteps between training info
            loggings
    """

    def __init__(
        self,
        agent: Any,
        env: gym.Env,
        root: Optional[str] = None,
        max_timesteps: int = 2 ** 20,
        stop_if_reach_goal: bool = True,
        eval_episodes: int = 4,
        print_gap: int = 2 ** 11
    ) -> None:
        self.env = env
        self.eval_env = deepcopy(self.env)
        self.agent = agent

        if root:
            self.root = os.path.expanduser(root)
        else:
            self.root = None
        self.eval_episodes = eval_episodes
        self.print_gap = print_gap
        self.max_timesteps = max_timesteps
        self.stop_if_reach_goal = stop_if_reach_goal

        self.timestep = 0
        self.max_eval_reward = float('-inf')
        self.timestamp = get_datetime()
        self.last_print_step = 0

    def save(self, prefix: Optional[str] = None) -> None:
        """
        Save model weights and relevant hyperparameters of a given agent

        Args:
            step: Timestep at which model is being saved
            prefix (str, optional): Prefix of the save name
        """
        algo_name = self.agent.__class__.__name__
        env_name = self.env.name

        name = "{}_{}_{}".format(algo_name, env_name, self.timestamp)
        if prefix is not None:
            name = prefix + name

        path = os.path.join(self.root, name)
        mkdir(path)

        weights = self.agent.get_save()
        checkpoint_path = os.path.join(path, "agent-{}.pt".format(self.timestep))
        torch.save(weights, checkpoint_path)

    @abstractmethod
    def train(self) -> None:
        """Training method, to be defined in inherited classes"""
        pass

    @torch.no_grad()
    def evaluate(self) -> bool:
        """
        Evaluate performance of agent

        Returns:
            is_reach_goal (bool): Whether the agent reaches the target reward
        """
        reward_list = [self.get_episode_return() for _ in range(self.eval_episodes)]
        avg_reward = np.average(reward_list)
        std_reward = float(np.std(reward_list))

        if avg_reward > self.max_eval_reward:
            self.max_eval_reward = avg_reward
            if self.root:
                self.save()

        is_reach_goal = bool(self.max_eval_reward > self.env.target_reward)
        if is_reach_goal:
            print(
                f"Target Reward: {self.env.target_reward:8.2f}\t"
                f"Step: {self.timestep:8.2e}\t"
                f"Average Reward: {avg_reward:8.2f}\t"
                f"Std Reward: {std_reward:8.2f}\t"
            )

        if self.timestep - self.last_print_step > self.print_gap:
            self.last_print_step = self.timestep
            print(
                f"Step: {self.timestep:8.2e}\t"
                f"Average Reward: {avg_reward:8.2f}\t"
                f"Std Reward: {std_reward:8.2f}\t"
                f"Max Reward: {self.max_eval_reward:8.2f}\t"
            )

        return is_reach_goal

    @torch.no_grad()
    def get_episode_return(self) -> float:
        episode_return = 0.0
        state = self.eval_env.reset()
        for _ in range(self.eval_env.max_step):
            action = self.select_action(state)
            state, reward, done, _ = self.eval_env.step(action)
            episode_return += reward
            if done:
                break
        return self.eval_env.episode_return if hasattr(self.eval_env, 'episode_return') else episode_return
