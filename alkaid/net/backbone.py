import torch
import torch.nn as nn
from typing import Union, Optional, Sequence, List

def linearblock(
    in_dim: int,
    out_dim: int,
    activation: Optional[nn.Module] = None,
) -> List[nn.Module]:
    module_list = [nn.Linear(in_dim, out_dim)]
    if activation is not None:
        module_list += [activation]
    return module_list


class MLP(nn.Module):
    """
    A simple MLP backbone. Size of each layer is: ``(hidden_size[0], hidden_size[1])``,
    ``(hidden_size[1], hidden_size[2])``, ..., ``(hidden_size[-2], hidden_size[-1])``.

    Parameters
    ----------
    hidden_size : Sequence[int]
        A list of sizes for linear layers

    activation : Union[nn.Module, Sequence[nn.Module]], optional, default=nn.ReLu()
        Activation function(s) after each layer. You can pass an activation
        function to be used for all layers, or a list of activation functions
        for different layers. ``None`` to no activation.

    activ_last_layer : bool, optional, default=False
        Place an activation function after the last linear layer or not.
    """

    def __init__(
        self,
        hidden_size: Sequence[int],
        activation: Union[nn.Module, Sequence[nn.Module]] = nn.ReLU(),
        activ_last_layer: bool = False
    ) -> None:
        super(MLP, self).__init__()

        n_activ = len(hidden_size) - 1 if activ_last_layer else len(hidden_size) - 2
        if activation:
            if isinstance(activation, list):
                assert len(activation) == n_activ
                activation_list = activation
            else:
                activation_list = [activation for _ in range(n_activ)]
        else:
            activation_list = [None] * n_activ

        module_list = []
        for i in range(len(hidden_size) - 1):
            in_size, out_size = hidden_size[i], hidden_size[i+1]
            if i > len(activation_list) - 1:
                activ = None
            else:
                activ = activation_list[i]
            module_list += linearblock(in_size, out_size, activ)

        self.out_dim = hidden_size[-1]
        self.core = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.core(x)
        return output
