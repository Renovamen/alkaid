from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np
import torch
from torch import nn
import gym

class Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        soft_update_tau: float = 2 ** -8,
        device: Optional = None
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.env = env
        self.gamma = gamma

        self.timestep = 0
        self.soft_update_tau = soft_update_tau

    def set_timestep(self, timestep: int) -> None:
        """
        Update the current timestep before selecting an action

        Args:
            timestep (int): Ccurrent timestep
        """
        self.timestep = timestep

    def soft_target_update(self, current: nn.Module, target: nn.Module) -> None:
        """
        Update the target Q model using soft target update:

        .. math::
            \\theta' = \\tau \\theta' + (1 - \\tau) \\theta

        where :math:`\\theta'` is the parameters of the target model,
        :math:`\\theta` is the parameters of the current model.

        Args:
            current (nn.Module): The current model
            target (nn.Module): The target model
        """
        for target_p, current_p in zip(target.parameters(), current.parameters()):
            target_p.data.copy_(
                current_p.data.mul_(self.soft_update_tau) +
                target_p.data.mul_(1 - self.soft_update_tau)
            )

    @abstractmethod
    def get_save(self) -> Any:
        pass

    @abstractmethod
    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """
        Select an action using epsilon-greedy for exploration.

        Args:
            state (np.ndarray): Current state of the environment

        Returns:
            action (np.ndarray): Action taken by the agent
        """
        pass
