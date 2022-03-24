"""
Controllers that take actions in various forms along with states, and
returns action dictionaries suitable for use with the carla environment.
"""
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class BaseController(ABC):
    """Base Controller Class."""
    def __init__(self, controller_config):
        pass

    def _generate_action_dict(self, accel, steer):
        action_dict = {
            "throttle": np.clip(accel, 0, 1),
            "brake": abs(np.clip(accel, -1, 0)),
            "steer": np.clip(steer, -1, 1),
        }
        return action_dict

    @abstractmethod
    def get_control_dict(self, state, action):
        pass


class DummyController(BaseController):
    """Dummy controller that simply turns raw actions into a dictionary."""
    def __init__(self):
        pass

    def get_control_dict(self, state, action):
        accel = action[0, 0].item()
        steer = action[0, 1].item()
        action_dict = self._generate_action_dict(accel, steer)
        return action_dict


class LongitudinalPIDController(BaseController):
    """Take desired speed and raw steering as action inputs."""
    def __init__(self, controller_config):
        self._K_P = controller_config["kp"]
        self._K_I = controller_config["ki"]
        self._K_D = controller_config["kd"]
        self.window_size = controller_config["n"]

        self._window = deque([0 for _ in range(self.window_size)], maxlen=self.window_size)

    def get_control_dict(self, state, action):
        desired_speed = action[0, 0].item()
        steer = action[0, 1].item()
        error = desired_speed - state["speed"]
        accel = self._pid_step(error)
        action_dict = self._generate_action_dict(accel, steer)
        return action_dict

    def _pid_step(self, error):
        self._window.append(error)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        return self._K_P * error + self._K_I * integral + self._K_D * derivative


def get_controller(config) -> BaseController:
    controller_type = config["controller_type"]
    if controller_type == "dummy":
        controller = DummyController()
    elif controller_type == "longitudinal_pid":
        controller = LongitudinalPIDController(config["controller_config"])
    else:
        raise KeyError("Controller type: {} is invalid".format(controller_type))
    return controller
