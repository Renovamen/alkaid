from copy import deepcopy
from ..dqn import DQN

class DuelingDQN(DQN):
    def __init__(self, *args, **kwargs) -> None:
        super(DuelingDQN, self).__init__(*args, **kwargs)

        self.model.dueling = True
        self.model.make()
        self.target_model = deepcopy(self.model)
