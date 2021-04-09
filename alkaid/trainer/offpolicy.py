import os
import time
from copy import deepcopy
from typing import Optional
import numpy as np
import torch
import gym

from ..data.buffer import ReplayBuffer
from ..utils import get_datetime, mkdir


class OffPolicyTrainer:
    """
    A wrapper for off-policy training procedure.

    Off-policy agents: DQN (and its variants), DDPG, TD3, SAC

    Args:
        env (gym.Env): Environment
        agent: Agent
        max_timesteps (int, optional, default=2**20): Maximum limit of
            timesteps to train for
        batch_size (int, optional, default=2**8): Num of transitions sampled
            from replay buffer
        update_interval (int): Timesteps between network updates
        warmup_steps (int, optional, default=2**10): Number of warmup steps
            (take random actions to add randomness to training)
        stop_if_reach_goal (bool, optional, default=True): Stop when reach
            ``env.target_reward`` or not
        max_memo (int, optional, default=2**17): Capacity of replay buffer
        eval_times (int, optional, default=2**2)
        print_gap (int, optional, default=2**8)
    """

    def __init__(
        self,
        env: gym.Env,
        agent,
        root: Optional[str] = None,
        max_timesteps: int = 2 ** 20,
        batch_size: int = 2 ** 8,
        update_interval: int = 2 ** 10,
        warmup_steps: int = 2 ** 10,
        stop_if_reach_goal: bool = True,
        max_memo: int = 2 ** 17,
        eval_times: int = 2 ** 2,
        print_gap: int = 2 ** 6
    ):
        self.env = env
        self.eval_env = deepcopy(self.env)
        self.agent = agent
        self.agent.state = self.env.reset()

        self.root = os.path.expanduser(root)
        self.eval_times = eval_times
        self.print_gap = print_gap

        self.step = 0
        self.max_eval_reward = 0.
        self.timestamp = get_datetime()
        self.last_print_time = time.time()

        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.max_timesteps = max_timesteps
        self.stop_if_reach_goal = stop_if_reach_goal

        self.buffer = ReplayBuffer(
            capacity = max_memo + self.env.max_step,
            state_dim = self.env.state_dim,
            action_dim = 1 if self.env.is_discrete else self.env.action_dim,
            on_policy = False
        )

    @torch.no_grad()
    def warmup(self) -> None:
        state = self.env.reset()

        for step in range(self.warmup_steps):
            # sample an action randomly
            action = self.env.sample()
            # state transition
            next_state, reward, done, _ = self.env.step(action)

            # store state transition tuple to replay buffer
            mask = 0.0 if done else self.agent.gamma
            self.buffer.append(state, next_state, reward, mask, action)

            # update the current state
            state = self.env.reset() if done else next_state

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
        checkpoint_path = os.path.join(path, "agent-{}.pt".format(self.step))
        torch.save(weights, checkpoint_path)

    def train(self):
        self.warmup()

        # pre-training and hard update
        self.agent.update_params(self.buffer, self.warmup_steps, self.batch_size)

        self.step = self.warmup_steps

        if_reach_goal = False

        while self.step <= self.max_timesteps:
            with torch.no_grad():
                steps = self.agent.update_buffer(
                    buffer = self.buffer,
                    n_steps = self.update_interval
                )

            self.step += steps

            self.agent.update_params(
                buffer = self.buffer,
                n_steps = self.update_interval,
                batch_size = self.batch_size
            )

            with torch.no_grad():
                if_reach_goal = self.evaluate()

            if self.stop_if_reach_goal and if_reach_goal:
                break

    @torch.no_grad()
    def evaluate(self) -> bool:
        reward_list = [self.get_episode_return() for _ in range(self.eval_times)]
        avg_reward = np.average(reward_list)
        std_reward = float(np.std(reward_list))

        if avg_reward > self.max_eval_reward:
            print(avg_reward, self.max_eval_reward)
            self.max_eval_reward = avg_reward
            self.save()

        is_reach_goal = bool(self.max_eval_reward > self.env.target_reward)
        if is_reach_goal:
            print(
                f"Target Reward: {self.env.target_reward:8.2f}\t"
                f"Step: {self.step:8.2e}\t"
                f"Average Reward: {avg_reward:8.2f}\t"
                f"Std Reward: {std_reward:8.2f}\t"
            )

        if time.time() - self.last_print_time > self.print_gap:
            self.last_print_time = time.time()
            print(
                f"Step: {self.step:8.2e}\t"
                f"Average Reward: {avg_reward:8.2f}\t"
                f"Std Reward: {std_reward:8.2f}\t"
                f"Max Reward: {self.max_eval_reward:8.2f}\t"
            )

        return is_reach_goal

    def get_episode_return(self) -> float:
        episode_return = 0.0
        state = self.eval_env.reset()
        for _ in range(self.eval_env.max_step):
            state_tensor = torch.as_tensor((state,), device=self.agent.device)
            action_tensor = self.agent.actor(state_tensor)
            if self.eval_env.is_discrete:
                action_tensor = action_tensor.argmax(dim=1)
            action = action_tensor.cpu().numpy()[0]
            state, reward, done, _ = self.eval_env.step(action)
            episode_return += reward
            if done:
                break
        return self.eval_env.episode_return if hasattr(self.eval_env, 'episode_return') else episode_return
