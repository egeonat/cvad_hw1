from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseExplorer(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def generate_action(self):
        pass


class UniformExplorer(BaseExplorer):
    def __init__(self, config):
        action_space = torch.tensor(config["action_space"])
        self._action_sizes = action_space[:, 1] - action_space[:, 0]
        self._action_mins = action_space[:, 0]

    def generate_action(self, state):
        action = torch.rand((1, 2))
        action = action * self._action_sizes + self._action_mins
        return action


class SmartExplorer(BaseExplorer):
    """Smart explorer that only works with speed limits."""
    def __init__(self, config):
        self._speed_limit = 6.0
        self._controller_type = config["controller_type"]
        self._fps = config["fps"]

        self.accel_mean = None
        self.accel_counter = 0
        self.steer_mean = None
        self.steer_counter = 0

    def _generate_accel(self, state):
        """Every 1 to 3 seconds takes a noisy observation of the speed, and selects a new mean
        acceleration value to follow in a semi-intelligent manner."""
        speed = state["speed"] + np.random.randn() * 1.0
        if self.accel_counter == 0:
            self.accel_counter = np.random.randint(self._fps, self._fps * 3)
            if "pid" in self._controller_type:
                self.accel_mean = torch.rand((1, 1)) * 6
            else:
                if speed > self._speed_limit:
                    self.accel_mean = torch.rand((1, 1)) * (-1)
                elif speed < self._speed_limit:
                    self.accel_mean = torch.rand((1, 1))
        self.accel_counter -= 1
        accel = self.accel_mean + torch.randn_like(self.accel_mean) * 0.2
        if "pid" in self._controller_type:
            accel = torch.clamp(accel, 0.0, 6.0)
        else:
            accel = torch.clamp(accel, -1.0, 1.0)

        return accel

    def _generate_steer(self, state):
        command = state["command"]
        if self.steer_counter == 0:
            self.steer_counter = np.random.randint(self._fps, self._fps * 3)
            if command == 0:  # Left
                self.steer_mean = torch.randn((1, 1)) * 0.2 - 0.3
            elif command == 1:  # Right
                self.steer_mean = torch.randn((1, 1)) * 0.2 + 0.3
            elif command == 2 or command == 3:  # Straight & LaneFollow
                self.steer_mean = torch.randn((1, 1)) * 0.2
            elif command == 4:  # Change Lane Left
                self.steer_mean = torch.randn((1, 1)) * 0.2 - 0.1
            elif command == 5:  # Change Lane Right
                self.steer_mean = torch.randn((1, 1)) * 0.2 + 0.1
        self.steer_counter -= 1
        steer = self.steer_mean + torch.randn_like(self.steer_mean) * 0.2
        steer = torch.clamp(steer, -1.0, 1.0)
        return steer

    def generate_action(self, state):
        accel = self._generate_accel(state)
        steer = self._generate_steer(state)
        action = torch.cat((accel, steer), dim=1)
        return action


def get_explorer(config) -> BaseExplorer:
    explorer_type = config["explorer_type"]
    if explorer_type == "uniform":
        explorer = UniformExplorer(config)
    elif explorer_type == "smart":
        explorer = SmartExplorer(config)
    else:
        raise KeyError("Explorer type: {} is invalid".format(explorer_type))
    return explorer
