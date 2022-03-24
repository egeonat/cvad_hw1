"""Action augmenters are used to modify policy actions after the exploration stage."""
import random
from abc import ABC, abstractmethod

import torch


class BaseActionAugmenter(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def augment_action(self, action, state):
        pass


class DummyActionAugmenter(BaseActionAugmenter):
    """Simply passes the policy actions."""
    def augment_action(self, action, state):
        return action


class BrakeActionAugmenter(BaseActionAugmenter):
    """
    Randomly demonstrates braking actions based on the current state.
    Consistently follows actions for a few seconds before making a change.
    """
    def __init__(self, config):
        self._fps = config["fps"]
        self._controller_type = ["controller_type"]

        self._hazard_detected = False
        self._demonstrating = False
        self._steer_mean = None

    def augment_action(self, action, state):
        if not state["hazard"]:
            self._hazard_detected = False
            self._demonstrating = False
        elif state["hazard"] and not self._hazard_detected:
            self._hazard_detected = True
            self._demonstrating = random.random() <= 0.75
            self._steer_mean = action[:, 1:]

        if self._demonstrating:
            if "pid" in self._controller_type:
                accel = torch.rand((1, 1)) * 0.5
            else:
                accel = torch.rand((1, 1)) * (-1)
            steer = self._steer_mean
            steer += torch.randn_like(steer) * 0.1
            action = torch.cat((accel, steer), dim=1)
        return action


def get_augmenter(config) -> BaseActionAugmenter:
    augmenter_type = config["augmenter_type"]
    if augmenter_type == "brake_demo":
        augmenter = BrakeActionAugmenter(config)
    elif augmenter_type == "dummy":
        augmenter = DummyActionAugmenter(config)
    else:
        raise KeyError("Augmenter type: {} is invalid".format(augmenter_type))
    return augmenter
