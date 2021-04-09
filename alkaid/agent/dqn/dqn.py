import math
from typing import Any, Tuple, Optional
from copy import deepcopy
import gym
import numpy as np
import torch
from torch.nn import functional as F

class DQN:
    def __init__(
        self,
        # *args: Any,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        env: gym.Env,
        gamma: float = 0.99,
        eps_min: float = 0.01,
        eps_max: float = 1.0,
        eps_decay: int = 500,
        device: Optional = None
        # **kwargs: Any
    ) -> None:
        # super(DQN, self).__init__(*args, **kwargs)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.state = None

        self.critic = model
        self.critic_target = deepcopy(self.critic)
        self.actor = self.critic

        self.optim = optim

        self.gamma = gamma
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay = eps_decay

        self.step = 0
        self.soft_update_tau = 2 ** -8
        self.env = env

    @property
    def eps(self) -> float:
        """
        Calculate exploration rate for epsilon-greedy exploration after every timestep.
        """
        return self.eps_min + (self.eps_max - self.eps_min) * math.exp(
            -1.0 * self.step / self.eps_decay
        )

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """
        Select an action using epsilon-greedy for exploration.
        """
        self.step += 1

        if np.random.rand() < self.eps:
            action = self.env.sample()
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            q_value = self.actor(state.unsqueeze(0))
            action = q_value.squeeze().argmax(dim=-1).detach().cpu().numpy()
        return action

    def update_buffer(self, buffer, n_steps: int) -> int:
        """
        Store the state transition tuple to replay buffer.

        Args:
            buffer: Experience replay buffer
            n_steps (int): Explore ``n_steps`` steps in environment
        """
        for _ in range(n_steps):
            self.env.render()
            # select an action using epsilon-greedy
            action = self.select_action(self.state)
            # state transition
            next_state, reward, done, _ = self.env.step(action)
            # store state transition tuple to replay buffer
            mask = 0.0 if done else self.gamma
            buffer.append(self.state, next_state, reward, mask, action)
            # update the current state
            self.state = self.env.reset() if done else next_state
        return n_steps

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
                next_q_target_value = self.critic_target(next_state).max(dim=1, keepdim=True)[0]
                target_q_value = reward + mask * next_q_target_value
            q_value = self.critic(state).gather(1, action.type(torch.long))

            loss = F.mse_loss(q_value, target_q_value)

            # update model
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update target Q model
            self.soft_target_update()

    def soft_target_update(self) -> None:
        """
        Update the target Q model using soft target update:

        .. math::
            \\theta' = \\tau \\theta' + (1 - \\tau) \\theta

        where :math:`\\theta'` is the parameters of the target model,
        :math:`\\theta` is the parameters of the current model.
        """
        for target_p, current_p in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_p.data.copy_(
                current_p.data.__mul__(self.soft_update_tau) +
                target_p.data.__mul__(1 - self.soft_update_tau)
            )

    def get_save(self) -> Any:
        """Get model weights and relevant hyperparameters to be saved"""
        return self.actor.state_dict()
