import os
import numpy as np
import torch
import gym

from ..data import ReplayBuffer
from .base import Trainer


class OffPolicyTrainer(Trainer):
    """
    A wrap of off-policy training procedure.

    Off-policy agents: DQN (and its variants), DDPG, TD3, SAC

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

    batch_size : int, optional, default=2**8
        Num of transitions sampled from replay buffer

    update_interval : int
        Timesteps between network updates

    warmup_steps : int, optional, default=2**10
        Number of warmup steps (take random actions to add randomness to training)

    replay_size : int, optional, default=2**17
        Capacity of replay buffer

    stop_if_reach_goal : bool, optional, default=True
        Stop when reach ``env.target_reward`` or not

    eval_episodes : int, optional, default=4
        Number of episodes to evaluate for

    print_gap : int, optional, default=2**11
        Timesteps between training info loggings
    """

    def __init__(
        self,
        *args,
        batch_size: int = 2 ** 8,
        update_interval: int = 2 ** 10,
        warmup_steps: int = 2 ** 10,
        replay_size: int = 2 ** 17,
        **kwargs
    ):
        super(OffPolicyTrainer, self).__init__(*args, **kwargs)

        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.batch_size = batch_size

        self.gamma = self.agent.gamma
        self.buffer = ReplayBuffer(
            capacity = replay_size + self.env.max_step,
            state_dim = self.env.state_dim,
            action_dim = 1 if self.env.is_discrete else self.env.action_dim,
            on_policy = False
        )

        if self.ploter and self.ploter.x_scale == 1:
            self.ploter.x_scale = self.update_interval

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get the action to be performed on the environment

        For the first few timesteps (warmup steps) it selects an action randomly
        to introduce stochasticity to the environment start position.

        Parameters
        ----------
        state : np.ndarray
            Current state of the environment

        Returns
        -------
        action : np.ndarray
            Action to be taken
        """
        if self.timestep < self.warmup_steps:
            action = self.env.sample()
        else:
            action = self.agent.select_action(state)
        return action

    @torch.no_grad()
    def update_buffer(self, state: np.ndarray) -> np.ndarray:
        """
        Store the state transition tuple to replay buffer.

        Parameters
        ----------
        state : np.ndarray
            Current state of the environment

        Returns
        -------
        next_state : np.ndarray
            Next state
        """
        action = self.select_action(state)

        # state transition
        next_state, reward, done, _ = self.env.step(action)
        # store state transition tuple to replay buffer
        mask = 0.0 if done else self.gamma
        self.buffer.append(state, next_state, reward, mask, action)

        # update the current state
        state = self.env.reset() if done else next_state
        return state

    def train(self) -> None:
        state = self.env.reset()

        for self.timestep in range(0, self.max_timesteps):
            self.agent.set_timestep(self.timestep)

            # explore and store the state transition tuple to replay buffer.
            self.update_buffer(state)

            # render the current frame
            self.env.render()

            if self.timestep == self.warmup_steps:
                self.agent.update_params(self.buffer, self.warmup_steps, self.batch_size)
            elif (self.timestep > self.warmup_steps) \
                and (self.timestep - self.warmup_steps) % self.update_interval == 0:
                # update model parameters
                self.agent.update_params(self.buffer, self.update_interval, self.batch_size)

                # evaluate
                is_reach_goal = self.evaluate()

                # stop training after reaching the goal
                if self.stop_if_reach_goal and is_reach_goal:
                    break

        self.env.close()
