import math
from typing import Any, Tuple
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F

from ..base import Agent

class DQN(Agent):
    """
    Implementation of Deep Q-Network (DQN) proposed in [1].

    .. admonition:: References
        1. "`Playing Atari with Deep Reinforcement Learning. \
            <https://arxiv.org/abs/1312.5602>`_" Volodymyr Mnih, et al. arXiv 2013.
    """
    def __init__(
        self,
        *args,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        eps_min: float = 0.01,
        eps_max: float = 1.0,
        eps_decay: int = 500,
        **kwargs
    ) -> None:
        super(DQN, self).__init__(*args, **kwargs)

        self.model = model
        self.target_model = deepcopy(self.model)
        self.optim = optim

        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay

    @property
    def eps(self) -> float:
        """
        Calculate exploration rate for epsilon-greedy exploration after every timestep.
        """
        return self.eps_min + (self.eps_max - self.eps_min) * math.exp(
            -1.0 * self.timestep / self.eps_decay
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action using epsilon-greedy for exploration.

        Args:
            state (np.ndarray): Current state of the environment

        Returns:
            action (np.ndarray): Action taken by the agent
        """
        if np.random.rand() < self.eps:
            action = self.env.sample()
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            q_value = self.model(state.unsqueeze(0))
            action = q_value.squeeze().argmax(dim=-1).detach().cpu().numpy()
        return action

    def target_q_value(
        self, next_state: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # print('next_state: ', next_state.size())
        next_q_target_value = self.target_model(next_state)
        # print('next_q_target_value: ', next_q_target_value.size())
        max_next_q_target_value = next_q_target_value.max(dim=1, keepdim=True)[0]
        print('max_next_q_target_value: ', max_next_q_target_value.size())
        target_q_value = reward + mask * max_next_q_target_value
        return target_q_value

    def update_params(self, buffer, n_steps: int, batch_size: int) -> Tuple[float]:
        """
        Update parameters of the model using the sampled data from replay buffer.

        Args:
            buffer: Experience replay buffer
            n_steps (int): Explore ``n_steps`` steps in environment
            batch_size (int): Batch size of the data to be sampled from replay buffer
        """
        for _ in range(n_steps):
            with torch.no_grad():
                state, next_state, reward, mask, action = buffer.sample(batch_size)
                target_q_value = self.target_q_value(next_state, reward, mask)

            q_value = self.model(state).gather(1, action.type(torch.long))

            loss = F.mse_loss(q_value, target_q_value)

            # update model
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update target Q model
            self.soft_target_update(self.model, self.target_model)

    def get_save(self) -> Any:
        """Get model weights and relevant hyperparameters to be saved"""
        return self.model.state_dict()
