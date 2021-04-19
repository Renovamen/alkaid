import os
import numpy as np
from numbers import Number
from typing import Union, Optional

from .common import get_datetime, mkdir

class Logger:
    """
    A loggger to log and visualize (based on Tensorboard or ``alkaid.utils.Ploter``)
    statistics.

    Parameters
    ----------
    root : str
        Root directory to save log files

    log_basename : str, optional
        Base name of the log file

    tensorboard : bool, optional, default=False
        Enable tensorboard or not (``tensorboard`` package is required)

    verbose: bool, optional, default=True
    """

    def __init__(
        self,
        root: str,
        log_basename: Optional[str] = None,
        tensorboard: bool = False,
        verbose: bool = True
    ) -> None:
        self.root = os.path.expanduser(root)

        self.timestamp = get_datetime()
        self.log_basename = log_basename
        self.verbose = verbose

        self.text_path = os.path.join(self.root, 'text')
        mkdir(self.text_path)

        self.writter = None
        if tensorboard:
            self.tensorboard_path = os.path.join(self.root, 'tensorboard', self.log_name)
            mkdir(self.tensorboard_path)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writter = SummaryWriter(self.tensorboard_path)
            except ImportError:
                print(
                    "Warning: Tensorboard is configured to use, but currently not "
                    "installed on this machine. Please install Tensorboard with "
                    "'pip install tensorboard' or set ``tensorboard`` to ``False``."
                )

    @property
    def log_name(self) -> str:
        if self.log_basename and self.log_basename != '':
            return self.log_basename + '_' + self.timestamp
        else:
            return self.timestamp

    def write_tensorboard(
        self, key: str, x: Union[Number, np.number], y: Union[Number, np.number]
    ) -> None:
        """
        Log data into Tensorboard.

        Parameters
        ----------
        key : str
            Namespace which the input data tuple belongs to

        x : Union[Number, np.number]
            Ordinate of the input data

        y : Union[Number, np.number]
            Abscissa of the input data
        """
        self.writter.add_scalar(key, y, global_step=x)

    def write_text(self, text: str) -> None:
        """
        Log data into text files.

        Parameters
        ----------
        text : str
            A string to be logged
        """
        log_file_path = os.path.join(self.text_path, self.log_name + '.log')
        with open(log_file_path, "a") as f:
            f.write(text)

    def log(
        self, data: dict, step: int, addition: Optional[str] = None
    ) -> None:
        """
        Log statistics generated during updating.

        Parameters
        ----------
        data : dict
            Data to be logged

        step : int
            Step of the data to be logged

        addition : str, optional
            Additional information to be logged
        """
        text = f"step: {step:8.2e}\t"

        if addition:
            text = f"{addition}\t" + text

        for name, value in data.items():
            # log statistics to Tensorboard
            if self.writter is not None:
                self.write_tensorboard(name, step, value.recent)
            # log statistics to text files
            text += '{name}: {recent:7.2f}\t'.format(name=name, recent=value.recent)

        self.write_text(text + '\n')

        if self.verbose:
            print(text)

        self.last_log_step = step
