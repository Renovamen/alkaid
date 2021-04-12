from copy import deepcopy
from ..dqn import DQN

class DuelingDQN(DQN):
    """
    Implementation of Dueling DQN proposed in [1].

    The difference between Dueling DQN and DQN is in the network rather than here.

    .. admonition:: References
        1. "`Dueling Network Architectures for Deep Reinforcement Learning. \
            <https://arxiv.org/abs/1511.06581>`_" Ziyu Wang, et al. ICML 2016.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(DuelingDQN, self).__init__(*args, **kwargs)

        self.model.dueling = True
        self.model.make()
        self.target_model = deepcopy(self.model)
