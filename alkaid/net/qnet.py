import torch
import torch.nn as nn
from typing import Union, Sequence

from .backbone import MLP

class QNet(nn.Module):
    """
    A simple network for DQN and its variants.

    Parameters
    ----------
    state_dim : int
        Dimension of state space

    action_dim : int
        Dimension of action space

    hidden_size : Sequence[int]
        A list of sizes for all middle linear layers.

    activation : Union[nn.Module, Sequence[nn.Module]], optional, default=nn.ReLu()
        Activation function(s) after each layer. You can pass an activation
        function to be used for all layers, or a list of activation functions
        for different layers. ``None`` to no activation.

    softmax : bool, optional, default=False
        Apply a softmax over the last layer's output or not

    dueling : bool, optional, default=False
        Use dueling network or not (for Dueling DQN)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: Sequence[int] = (256, 256, 256),
        activation: Union[nn.Module, Sequence[nn.Module]] = nn.ReLU(),
        softmax: bool = False,
        dueling: bool = False
    ) -> None:
        super(QNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.softmax = softmax
        self.dueling = dueling
        self.hidden_size = [state_dim] + list(hidden_size) + [action_dim]

        self.make()

    def make(self) -> None:
        if self.dueling:
            self.core = MLP(self.hidden_size[:-1], self.activation, activ_last_layer=True)
            self.advantage = MLP([self.core.out_dim, self.action_dim], None)
            self.value = MLP([self.core.out_dim, 1], None)
        else:
            self.core = MLP(self.hidden_size, self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.core(x)

        if self.dueling:
            v, adv = self.value(logits), self.advantage(logits)
            logits = v + adv - adv.mean(dim=1, keepdim=True)

        if self.softmax:
            logits = torch.softmax(logits, dim=-1)

        return logits
