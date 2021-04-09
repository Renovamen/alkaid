import random
from typing import Optional, Tuple
import numpy as np
import torch

from ..utils import to_tensor, to_numpy

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        on_policy: bool,
        device: Optional = None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.capacity = capacity
        self.action_dim = action_dim
        self.state_dim = state_dim

        self._pointer = 0
        self._full = False

        if on_policy:
            stack_dim = state_dim * 2 + action_dim * 2 + 2
        else:
            stack_dim = state_dim * 2 + action_dim + 2

        self.stack = torch.empty((capacity, stack_dim), dtype=torch.float32, device=self.device)

    def append(self, *exp) -> None:
        """
        Append new experience to replay buffer.
        """
        self.stack[self._pointer] = to_tensor(self.pack(exp), device=self.device)
        self._pointer += 1

        if self._pointer >= self.capacity:
            self._full = True
            self._pointer = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        """
        Return randomly sampled experiences from replay buffer.
        """
        indices = torch.randint(len(self) - 1, size=(batch_size,), device=self.device)
        batch = self.stack[indices]
        return self.unpack(batch)

    def pack(self, input: Tuple) -> np.ndarray:
        state, next_state, reward, mask, action = input
        packed_input = np.concatenate(
            (state, next_state, to_numpy((reward, mask, action))), axis=0
        )
        return packed_input

    def unpack(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        return (
            input[:, 0 : self.state_dim],  # state
            input[:, self.state_dim : self.state_dim * 2],  # next state
            input[:, self.state_dim * 2 : self.state_dim * 2 + 1],  # reward
            input[:, self.state_dim * 2 + 1 : self.state_dim * 2 + 2],  # mask = 0.0 if done else gamma
            input[:, self.state_dim * 2 + 2 :]  # action
        )

    def __len__(self) -> int:
        return self.capacity if self._full else self._pointer
