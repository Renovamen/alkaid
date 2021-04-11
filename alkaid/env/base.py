from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np

class EnvWrapper(ABC):
    """Base class for all environment wrappers"""

    def __init__(self, env: Any, render: bool = True, verbose: bool = True) -> None:
        self.env = env
        self.verbose = verbose
        self.is_render = render

    @property
    @abstractmethod
    def name(self) -> str:
        """Return name of the environment."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return the dimention of the action space."""
        pass

    @property
    @abstractmethod
    def action_max(self) -> int:
        """Return the dimention of the action space."""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the dimention of the observation space."""
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Return whether the action space is discrete."""
        pass

    @property
    @abstractmethod
    def target_reward(self) -> int:
        pass

    @property
    @abstractmethod
    def max_step(self) -> int:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray]:
        """Step the environment through given action."""
        pass

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Sample an action from environment's action space."""
        pass

    @abstractmethod
    def seed(self, seed: int = None) -> None:
        """Set environment seed"""
        pass

    @abstractmethod
    def render(self) -> None:
        """Render environment"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets state of environment"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close environment"""
        pass

    def summary(self) -> Tuple:
        line = "{:>22}  {:>20}\n"

        summary_str  = "---------------------------------------------\n"
        summary_str += "       Summarization of the Environment      \n"
        summary_str += "=============================================\n"
        summary_str += line.format("Name", self.name)
        summary_str += line.format("Discrete Action Space", self.is_discrete)
        summary_str += line.format("Dim of State Space", self.state_dim)
        summary_str += line.format("Dim of Action Space", self.action_dim)
        summary_str += line.format("Max Action Value", self.action_max)
        summary_str += line.format("Max Step", self.max_step)
        summary_str += line.format("Target Reward", self.target_reward)
        summary_str += "=============================================\n"

        print(summary_str) if self.verbose else None

        return self.name, self.state_dim, self.action_dim, self.action_max, \
            self.is_discrete, self.target_reward, self.max_step
