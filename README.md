# Alkaid

Trying to implement some reinforcement learning algorithms with easy to read code, high speed and well performance.


&nbsp;

## Features

The supported algorithms currently include:

- DQN (Deep Q-Networks)
    - [DQN](https://arxiv.org/abs/1312.5602)
    - [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
    - [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)

Other features:

- **Modularity and Readability**: Implement agents, networks, replay buffers, etc as configurable modules and in a easy to read way
- **Logging and Visualization**: Easily logging training statistics and visualizing them using Tensorboard or `alkaid.utils.Ploter`


&nbsp;

## Installation

```bash
git clone https://github.com/Renovamen/alkaid.git
cd alkaid
python setup.py install
```

or

```bash
pip install git+https://github.com/Renovamen/alkaid.git --upgrade
```


&nbsp;

## License

Alkaid is MIT licensed, see the [LICENSE](LICENSE) file for more details.
