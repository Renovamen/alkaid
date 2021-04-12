import torch
from ..dqn import DQN

class DoubleDQN(DQN):
    """
    Implementation of Double Q-Learning proposed in [1].

    .. admonition:: References
        1. "`Deep Reinforcement Learning with Double Q-learning. \
            <https://arxiv.org/abs/1509.06461>`_" Hado van Hasselt, et al. AAAI 2016.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(DoubleDQN, self).__init__(*args, **kwargs)

    def target_q_value(
        self, next_state: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        next_q_value = self.model(next_state)
        next_best_action = next_q_value.argmax(dim=1, keepdim=True)

        next_q_target_value = self.target_model(next_state)
        max_next_q_target_value = next_q_target_value.gather(
            dim=1, index=next_best_action.type(torch.long)
        )
        target_q_value = reward + mask * max_next_q_target_value

        return target_q_value
