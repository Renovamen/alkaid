import os
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional

import gym
import numpy as np
import torch

from ..agent import Agent
from ..utils import MetricTracker, Logger, Ploter, get_datetime, mkdir

class Trainer(ABC):
    """
    Base class for all trainers.

    Parameters
    ----------
    agent: Agent
        Agent

    env : gym.Env
        Environment

    root : str
        Directory where the checkpoints and logs should be saved

    max_timesteps : int, optional, default=2**20
        Maximum limit of timesteps to train for

    stop_if_reach_goal : bool, optional, default=True
        Stop when reach ``env.target_reward`` or not

    eval_episodes : int, optional, default=4
        Number of episodes to evaluate for

    logger : Logger, optional
        An instance of :class:`~alkaid.utils.Logger` class

    ploter : Ploter, optional
        An instance of :class:`~alkaid.utils.Ploter` class
    """

    def __init__(
        self,
        agent: Agent,
        env: gym.Env,
        root: Optional[str] = None,
        max_timesteps: int = 2 ** 20,
        stop_if_reach_goal: bool = True,
        eval_episodes: int = 4,
        logger: Optional[Logger] = None,
        ploter: Optional[Ploter] = None
    ) -> None:
        self.env = env
        self.eval_env = deepcopy(self.env)
        self.agent = agent

        if root:
            self.root = os.path.expanduser(root)
        else:
            self.root = None

        self.eval_episodes = eval_episodes
        self.max_timesteps = max_timesteps
        self.stop_if_reach_goal = stop_if_reach_goal

        self.timestep = 0
        self.timestamp = get_datetime()

        self.tracker = MetricTracker('test/rew', 'test/rew_std')

        self.logger = logger

        if self.logger and self.logger.log_basename is None:
            self.logger.log_basename = self.basename

        self.ploter = ploter

        if self.ploter:
            self.ploter.add_line(self.basename)
            if self.ploter.save_name is None:
                self.ploter.save_name = self.basename

    @property
    def basename(self) -> str:
        algo_name = self.agent.__class__.__name__
        env_name = self.env.name
        return "{}_{}".format(algo_name, env_name)

    def save(self, prefix: Optional[str] = None) -> None:
        """
        Save model weights and relevant hyperparameters of a given agent

        Parameters
        ----------
        prefix : str, optional
            Prefix of the save name, ``None`` if no prefix
        """
        name = "{}_{}".format(self.basename, self.timestamp)
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

        Returns
        -------
        is_reach_goal : bool
            Whether the agent reaches the target reward
        """
        reward_list = [self.get_episode_return() for _ in range(self.eval_episodes)]
        avg_reward = np.average(reward_list)
        std_reward = np.std(reward_list)

        if avg_reward > self.tracker['test/rew'].max and self.root:
            self.save()

        self.tracker.update('test/rew', avg_reward)
        self.tracker.update('test/rew_std', std_reward)

        is_reach_goal = bool(self.tracker['test/rew'].max > self.env.target_reward)
        if is_reach_goal and self.logger is not None:
            self.logger.log(
                data = self.tracker.metrics,
                step = self.timestep,
                force = True,
                addition = f"READCH GOAL! Target Reward: {self.env.target_reward:8.2f}"
            )

        if self.logger is not None:
            self.logger.log(data=self.tracker.metrics, step=self.timestep)

        if self.ploter is not None:
            self.ploter.append([np.min(reward_list), avg_reward, np.max(reward_list)])
            self.ploter.plot()

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
