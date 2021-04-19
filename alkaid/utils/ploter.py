# inspired by: https://zhuanlan.zhihu.com/p/75477750

import os
from numbers import Number
from typing import Optional
import numpy as np

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
        ``alkaid.trainer.Trainer``.

    title : str, optional, default=''
        Title of the figure

    title_size : int, optional, default=20
        Font size of the title

    label_x : str, optional, default='Time Step'
        Label of the X-axis

    label_y : str, optional, default='Reward'
        Label of the Y-axis

    label_size : int, optional, default=15
        Font size of the X-axis and Y-axis labels

    x_scale : Number, optional, default=1
        Scale of the X-axis. For example, you plot a point every 1000 steps, then
        your ``x_scale`` may be ``1000``. This can be automatically set by
        ``alkaid.trainer.Trainer``.
    """
    def __init__(
        self,
        root: str = None,
        save_name: Optional[str] = None,
        title: str = '',
        title_size: int = 20,
        label_x: str = 'Time Step',
        label_y: str = 'Reward',
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

    def set_x_scale(self, x_scale: Number):
        self._x_scale = x_scale

    def add_line(self, name: str, style: str = None) -> None:
        self._lines.append([])
        self._line_names.append(name)
        self._line_styles.append(style)

    def to_index(self, name: Optional[str] = None) -> int:
        if name:
            return self._line_names.index(name)
        else:
            return 0

    def append(self, data: list, name: Optional[str] = None):
        index = self.to_index(name)

        for (i, val) in enumerate(data):
            if i < len(self._lines[index]):
                self._lines[index][i].append(val)
            else:
                self._lines[index].append([val])

    def plot(self) -> None:
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            import seaborn as sns
            sns.set()
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
