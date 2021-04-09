from numbers import Number
from typing import Union, Optional
import numpy as np
import torch

def to_numpy(
    x: Union[list, tuple, np.ndarray, torch.Tensor, np.number, Number]
) -> Union[list, tuple, np.ndarray]:
    """
    Convert an object to np.ndarray.

    Args:
        x(Union[list, tuple, np.ndarray, torch.Tensor]): Object need to be converted
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (np.number, Number, list, tuple)):
        return np.asanyarray(x)
    else:
        raise TypeError(f"Converting {type(x)} to numpy.ndarray is not supported.")

def to_tensor(
    x: Union[list, tuple, np.ndarray, torch.Tensor, np.number, Number],
    dtype: Optional[torch.dtype] = None,
    device: Union[str, int, torch.device] = "cpu"
) -> Union[dict, list, tuple, torch.Tensor]:
    """
    Convert an object to torch.Tensor.

    Args:
        x(Union[list, tuple, np.ndarray, torch.Tensor]): Object need to be converted
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    elif isinstance(x, (np.number, Number, list, tuple)):
        return to_tensor(np.asanyarray(x), dtype, device)
    else:
        raise TypeError(f"Converting {type(x)} to torch.Tensor is not supported.")
