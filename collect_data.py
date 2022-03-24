import json
import os

import cv2
import torch
import yaml
from agents.navigation.controller import VehiclePIDController

from carla_env.env import Env
from rl.tools.visualizer import RLVisualizer
from utils.noiser import Noiser


class DataCollector():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        os.makedirs(os.path.join(config["data_dir"], "measurements"), exist_ok=True)
        os.makedirs(os.path.join(config["data_dir"], "rgb"), exist_ok=True)

        self._args_lateral_dict = {'K_P': 0.5, 'K_I': 0.1, 'K_D': 0.0, 'dt': 1/20}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': 1/20}

        self.noiser = Noiser(config["fps"])
        self.visualizer = RLVisualizer(config)
        self.saved_samples = 0
        self.step = 0
        self.prev_command = 3
        self.cur_command = 3

    def reset_env(self):
        state, _, is_terminal = self.env.reset()
        self.prev_command = 3
        self.cur_command = 3
        self.autopilot = VehiclePIDController(self.env._ego_vehicle,
                                              self._args_lateral_dict,
                                              self._args_longitudinal_dict,
                                             )
        return state

    def save_step(self, state, action):
        measurements = {
            "speed": state["speed"],
            "throttle": action["throttle"],
            "brake": action["brake"],
            "steer": action["steer"],
            "command": state["command"],
            "route_dist": state["route_dist"],
            "route_angle": state["route_angle"],
            "lane_dist": state["lane_dist"],
            "lane_angle": state["lane_angle"],
            "hazard": state["hazard"],
            "hazard_dist": state["hazard_dist"],
            "tl_state": state["tl_state"],
            "tl_dist": state["tl_dist"],
            "is_junction": state["is_junction"]
        }
        json_path = os.path.join(self.config["data_dir"],
                                 "measurements",
                                 f"{self.saved_samples:08}.json")
        with open(json_path, "w") as f:
            json.dump(measurements, f)

        rgb = state["rgb"]
        rgb_path = os.path.join(self.config["data_dir"], "rgb", f"{self.saved_samples:08}.png")
        cv2.imwrite(rgb_path, rgb)
        self.saved_samples += 1

    def take_step(self, state):
        target_speed = state["optimal_speed"] * 3.6
        target_wp_ind = self.env._plan_manager.dense_wp_ind
        target_wp = self.env._plan_manager.dense_plan[target_wp_ind][0]
        control = self.autopilot.run_step(target_speed, target_wp)
        action = {
            "throttle": control.throttle,
            "brake": control.brake,
            "steer": control.steer
        }
        if self.step % 10 == 0:
            self.save_step(state, action)
        action["steer"] = self.noiser.tick(action["steer"])
        self.step += 1
        state, reward_dict, is_terminal = self.env.step(action)
        self.visualizer.visualize(state, torch.Tensor([0.0, 0.0]), reward_dict)
        return state, is_terminal

    def collect(self, num_samples):
        state = self.reset_env()
        for i in range(num_samples):
            state, is_terminal = self.take_step(state)
            if is_terminal:
                self.reset_env()


def main():
    with open(os.path.join("configs", "data_collect.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        collector = DataCollector(env, config)
        collector.collect(1000000)


if __name__ == "__main__":
    main()
