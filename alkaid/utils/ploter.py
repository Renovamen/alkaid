# inspired by: https://zhuanlan.zhihu.com/p/75477750

import os
from numbers import Number
from typing import Optional
import numpy as np
import pickle

from .common import get_datetime, mkdir

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

    title : str, optional, default=''
        Title of the figure

    title_size : int, optional, default=20
        Font size of the title

    label_x : str, optional, default='Time Step'
        Label of the X-axis

    label_y : str, optional, default='Score'
        Label of the Y-axis

    label_size : int, optional, default=15
        Font size of the X-axis and Y-axis labels

    x_scale : Number, optional, default=1
        Scale of the X-axis. For example, you plot a point every 1000 steps, then
        your ``x_scale`` may be ``1000``. This can be automatically set by
        :func:`alkaid.trainer.Trainer`.
    """
    def __init__(
        self,
        root: str,
        save_name: Optional[str] = None,
        title: str = '',
        title_size: int = 20,
        label_x: str = 'Time Step',
        label_y: str = 'Score',
        label_size: int = 15,
        x_scale: Number = 1
    ) -> None:
        self.root = os.path.expanduser(root)
        mkdir(self.root)

        self.save_name = save_name
        self.timestamp = get_datetime()

        self._title = title
        self._title_size = title_size
        self._label_x = label_x
        self._label_y = label_y
        self._label_size = label_size
        self._x_scale = x_scale

        self.clear()

    def clear(self) -> None:
        self._lines = []
        self._line_names = []
        self._line_styles = []

    def set_title(self, title: str, font_size: int = 20) -> None:
        self._title = title
        self._title_size = font_size

    def set_label(self, x: str, y: str, font_size: int = 15) -> None:
        self._label_x = x
        self._label_y = y
        self._label_size = font_size

    def set_x_scale(self, x_scale: Number) -> None:
        self._x_scale = x_scale

    def add_line(self, name: str, data: list, style: str = None) -> None:
        if name in self._line_names:
            index = self._to_index(name)
            self._lines[index] = data
            self._line_styles[index] = style
        else:
            self._lines.append(data)
            self._line_names.append(name)
            self._line_styles.append(style)

    def _to_index(self, name: str) -> int:
        return self._line_names.index(name)

    def plot(self) -> None:
        """Visualize training results in a figure."""
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            import seaborn as sns
            sns.set(style='whitegrid')
        except ImportError:
            print(
                "Warning: 'alkaid.utils.Ploter' requires pandas, matplotlib and "
                "seaborn, some of which are currently not installed on this machine. "
                "Please install them with 'pip install pandas matplotlib seaborn'."
            )

        df = []

        for i, line in enumerate(self._lines):
            line = np.array(line)
            df.append(
                pd.DataFrame(line).melt(
                    var_name = self._label_x,
                    value_name = self._label_y
                )
            )
            df[i][self._label_x] *= self._x_scale
            df[i]['label'] = self._line_names[i]
            df[i]['lineStyle'] = self._line_styles[i]

        data = pd.concat(df)

        figure = sns.lineplot(
            data = data,
            x = self._label_x,
            y = self._label_y,
            hue = 'label',
            style = 'label',
            markers = self._line_styles
        )
        figure.legend_.set_title(None)

        plt.xlabel(self._label_x)
        plt.ylabel(self._label_y)
        plt.title(self._title, fontsize=self._title_size)

        name = f"{self.save_name if self.save_name else 'figure'}_{self.timestamp}"
        plt.savefig(os.path.join(self.root, name + '.jpg'))

        plt.close()

    def load_from_pkl(self, log_dir: str):
        """
        Load the ``pkl`` format data logged by :func:`alkaid.utils.Logger`. This
        is useful when you want to plot results of different agents in a same figure.

        Parameters
        ----------
        log_dir: str
            Path to the log directory. All ``.pkl`` files under this path will be
            loaded, file name of each will be served as its corresponding line label.
        """
        self.clear()

        for root, _, files in os.walk(log_dir):
            for fname in files:
                if not fname.endswith('pkl'):
                    continue

                fpath = os.path.join(root, fname)
                with open(fpath, 'rb') as f:
                    data = pickle.loads(f.read())

                self.add_line(
                    name = fname[:-4],
                    data = [data['test/rew'], data['test/rew_min'], data['test/rew_max']]
                )
