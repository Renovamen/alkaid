from typing import Any, Tuple, Optional
import gym
import numpy as np

class GymWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        render: bool = True,
        verbose: bool = True
    ):
        super(GymWrapper, self).__init__(env)

        self.env = env
        self.verbose = verbose
        self.is_render = render

        self.mean = mean
        self.std = std

        self._eps = np.finfo(np.float32).eps.item()
        self.summary()

    def __getattr__(self, name: str) -> Any:
        """All other calls would go to self.env"""
        env = super(GymWrapper, self).__getattribute__("env")
        return getattr(env, name)

    @property
    def name(self) -> str:
        return self.env.unwrapped.spec.id

    @property
    def action_dim(self) -> int:
        """Return the dimention of the action space."""
        if self.is_discrete:  # discrete action space
            return self.env.action_space.n
        else:  # continuous action space
            return self.env.action_space.shape[0]

    @property
    def action_max(self) -> int:
        """Return the dimention of the action space."""
        if self.is_discrete:  # discrete action space
            return int(1)
        else:  # continuous action space
            return float(self.env.action_space.high[0])

    @property
    def state_dim(self) -> int:
        """Return the dimention of the observation space."""
        if isinstance(self.env.observation_space, gym.spaces.Discrete):  # discrete observation space
            return self.env.observation_space.n
        else:  # continuous observation space
            return self.env.observation_space.shape[0]

    @property
    def is_discrete(self) -> bool:
        """Return whether the action space is discrete."""
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def target_reward(self) -> int:
        return getattr(self.env, 'target_reward', None) or getattr(
            self.env.spec, 'reward_threshold', None) or 2 ** 16

    @property
    def max_step(self) -> int:
        return getattr(self.env, 'max_step', None) or getattr(
            self.env, '_max_episode_steps', None) or 2 ** 10

    def reset(self) -> np.ndarray:
        """
        Reset environment

        Returns:
            state(np.ndarray): Initial state
        """
        state = self.env.reset()
        return self.norm_obs(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray]:
        """
        Step the environment through given action

        Args:
            action(np.ndarray): Action taken by agent in the current step

        Returns:
            The next state, reward, game status (end or not) and debugging info
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        return self.norm_obs(state), reward, done, info

    def norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.mean and self.std:
            clip_max = 10.0  # from stable-baselines3, see: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py
            obs = (array - self.mean) / (self.std + self._eps)
            obs = np.clip(obs, -clip_max, clip_max)
        return obs

    def seed(self, seed: int = None) -> None:
        """
        Set environment seed

        Args:
            seed(int): Seed value
        """
        self.env.seed(seed)

    def sample(self) -> np.ndarray:
        """
        Sample an action from environment's action space.

        Returns:
            action(np.ndarray): Random action from action space
        """
        return self.env.action_space.sample()

    def render(self, **kwargs: Any) -> None:
        """Render environment"""
        if self.is_render:
            self.env.render(**kwargs)

    def close(self) -> None:
        """Closes environment"""
        self.env.close()

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
