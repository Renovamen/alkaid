import os
from typing import Optional, Tuple, Union
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .common import get_datetime, mkdir

COLOR_LIST = [
	'blue',
    'green',
    'red',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'purple',
    'pink',
	'brown',
    'orange',
    'teal',
    'lightblue',
    'lime',
    'lavender',
    'turquoise',
	'darkgreen',
    'tan',
    'salmon',
    'gold',
    'darkred',
    'darkblue'
]

def smooth(data: list, window: int, mode='single'):
    """
    Smooths values using a sliding window.

    Parameters
    ----------
    data : list
        Data to be smoothed

    window : int
        Size of the kernel window

    mode : str, optional, default='single'
        'single' / 'both'. If mode = 'single', average the window [i - window, i].
        If mode = 'both', average the window [i - window, i + window].
    """
    assert mode in ('both', 'single')

    data = np.asarray(data)

    if window <= 1:
        return data

    if mode == 'both':
        if len(data) < 2 * window + 1:
            return np.ones_like(data) * data.mean()
        kernel = np.ones(2 * window + 1)
    elif mode == 'single':
        if window > len(data):
            return np.ones_like(data) * data.mean()
        kernel = np.ones(window)

    out = np.convolve(data, kernel, mode='same') / np.convolve(np.ones_like(data), kernel, mode='same')

    return out


class Ploter:
    """
    Visualize training statistics using matplotlib and seaborn.ArithmeticError

    Parameters
    ----------
    root : str
        Root directory to save figure

    save_name : str, optional
        Base name of the saved figure. This can be automatically set by
        :func:`alkaid.trainer.Trainer`.

    title : str, optional
        Title of the figure

    title_size : Union[float, str], optional, default=20
        Font size of the title

    label_x : str, optional, default='Time Step'
        Label of the X-axis

    label_y : str, optional, default='Score'
        Label of the Y-axis

    label_size : Union[float, str], optional, default=15
        Font size of the X-axis and Y-axis labels

    figsize : Tuple[float], optional, default=(6.4, 4.8)
        Size of the plotted figure, a tuple containing width and height in inches

    x_scale : int, optional, default=1
        Scale of the X-axis. For example, you plot a point every 1000 steps, then
        your ``x_scale`` may be ``1000``. This can be automatically set by
        :func:`alkaid.trainer.Trainer`.
    """
    def __init__(
        self,
        root: str,
        save_name: Optional[str] = None,
        title: Optional[str] = None,
        title_size: Union[float, str] = 20,
        label_x: str = 'Time Steps',
        label_y: str = 'Score',
        label_size: Union[float, str] = 15,
        figsize: Tuple[float] = (6.4, 4.8),
        x_scale: int = 1
    ) -> None:
        self.root = os.path.expanduser(root)
        mkdir(self.root)

        self.save_name = save_name
        self.timestamp = get_datetime()

        self.title = title
        self.title_size = title_size
        self.label_x = label_x
        self.label_y = label_y
        self.label_size = label_size
        self.x_scale = x_scale

        self.figsize = figsize

        self.clear()

    def clear(self) -> None:
        self.lines = {}

    def add_line(
        self, name: str, mean: list, min_bound: Optional[list] = None, max_bound: Optional[list] = None
    ) -> None:
        self.lines[name] = dict(mean=mean, min=min_bound, max=max_bound)

    def plot(self, smooth_window: int = 0, smooth_mode: str = 'single') -> None:
        """
        Visualize training results in a figure.

        Parameters
        ----------
        smooth_window : int
            Size of the kernel window

        smooth_mode : str, optional, default='single'
            'single' / 'both'. If mode = 'single', average using window [i - smooth_window, i].
            If mode = 'both', average using window [i - smooth_window, i + smooth_window].
        """

        fig, ax = plt.subplots(figsize=self.figsize)

        # lines
        for i, (agent, data) in enumerate(self.lines.items()):
            x = np.arange(len(data['mean'])) * self.x_scale
            color = COLOR_LIST[i % len(COLOR_LIST)]

            # plot intervals between min and max bound
            if data['min'] is not None and data['max'] is not None:
                y_min = smooth(data['min'], smooth_window, smooth_mode)
                y_max = smooth(data['max'], smooth_window, smooth_mode)
                ax.fill_between(x, y_min, y_max, color=color, alpha=0.1, lw=0)

            # plot means
            y = smooth(data['mean'], smooth_window, smooth_mode)
            ax.plot(x, y, c=color, label=agent, alpha=0.5, lw=1)

        # legend
        ax.legend()

        # label for x-axis and y-axis
        ax.set_xlabel(self.label_x)
        ax.set_ylabel(self.label_y)

        # title
        if self.title:
            ax.title(self.title, fontsize=self.title_size)

        # grids
        ax.grid(linewidth=0.5, alpha=0.5)

        # box styles
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)

        # save figure
        name = f"{self.save_name if self.save_name else 'figure'}_{self.timestamp}"
        plt.savefig(os.path.join(self.root, name + '.jpg'))

        plt.close()

    def load_data(self, data_dir: str):
        """
        Load the ``pkl`` format data logged by :func:`alkaid.utils.Logger`. This
        is useful when you want to plot results of different agents in a same figure.

        Parameters
        ----------
        data_dir: str
            Path to the log directory. All ``.pkl`` files under this path will be
            loaded, file name of each will be served as its corresponding line label.
        """
        self.clear()

        for root, _, files in os.walk(data_dir):
            for fname in files:
                if not fname.endswith('pkl'):
                    continue

                fpath = os.path.join(root, fname)
                with open(fpath, 'rb') as f:
                    data = pickle.loads(f.read())

                self.add_line(
                    name = fname[:-4],
                    mean = data['test/rew'],
                    min_bound = data['test/rew_min'],
                    max_bound = data['test/rew_max'],
                )
