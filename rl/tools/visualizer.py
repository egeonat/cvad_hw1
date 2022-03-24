from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.mappings import ROAD_OPTIONS

WHITE = (255, 255, 255)


def labels_to_cityscapes_palette(array):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [70, 70, 200],
        1: [190, 153, 153],
        2: [72, 0, 90],
        3: [220, 20, 60],
    }
    result = np.zeros((array.shape[1], array.shape[2], 3), np.uint8)
    for key, value in classes.items():
        result[np.where(array[3-key, :, :] >= 0.4)] = value
    return result


class BaseVisualizer(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def visualize(self, state, action, reward_dict):
        pass


class DummyVisualizer(BaseVisualizer):
    def visualize(self, *args):
        pass


class RLVisualizer(BaseVisualizer):
    def __init__(self, config):
        if config["feature_extractor_type"] == "bev":
            self.img_size = 256
        elif config["feature_extractor_type"] == "lss":
            self.img_size = 256
        elif config["feature_extractor_type"] == "max_pool":
            self.img_size = 256
        else:
            self.img_size = 288
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.border_w = 400
        self.font_scale = 0.8
        self.text_left_pad = 5
        self.text_top_pad = 20
        self.text_interval = 25

        self.text_inds = [0, 0]

        self.last10_actions = np.zeros((10, 2))
        self.counter = 0

        self.hazard_fig = plt.figure()
        self.hazard_plt = self.hazard_fig.add_subplot()
        self.hazard_line, = self.hazard_plt.fill([], [])

    def write_text(self, img, text, side):
        if side == "l":
            text_count = self.text_inds[0]
            self.text_inds[0] += 1
            x_pos = self.text_left_pad
        elif side == "r":
            text_count = self.text_inds[1]
            self.text_inds[1] += 1
            x_pos = self.text_left_pad + self.border_w + self.img_size
        y_pos = self.text_top_pad + text_count * self.text_interval
        img = cv2.putText(img, text, (x_pos, y_pos), self.font, self.font_scale, WHITE)
        return img

    def draw_hazard_zones(self, coords):
        v_coords, w_coords, ego_coords = coords
        self.hazard_plt.cla()
        for xs, ys in v_coords:
            self.hazard_plt.fill(xs, ys, alpha=0.75, fc="r")
        for xs, ys in w_coords:
            self.hazard_plt.fill(xs, ys, alpha=0.75, fc="b")
        xs, ys = ego_coords
        self.hazard_plt.fill(xs, ys, alpha=0.5, fc="g")
        plt.pause(0.01)

    def visualize(self, state, action, reward_dict):
        if "semantic" in state:
            img = labels_to_cityscapes_palette(state["semantic"])
        if "semantic_bev" in state:
            img = labels_to_cityscapes_palette(state["semantic_bev"])
        elif "bev" in state:
            img = labels_to_cityscapes_palette(state["bev"])
            img = np.pad(img, ((0, 128), ((0, 0)), (0, 0)))
        else:
            img = state["rgb"]
        action_size = action.shape[-1]
        self.last10_actions[self.counter % 10, :action_size] = action.numpy()
        avg_action = np.average(self.last10_actions, axis=0)
        self.text_inds = [0, 0]
        self.counter += 1

        vis_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vis_img = cv2.resize(vis_img, (self.img_size, self.img_size))
        vis_img = cv2.copyMakeBorder(
            vis_img, 0, 0, self.border_w, self.border_w, cv2.BORDER_CONSTANT)

        vis_img = self.write_text(
            vis_img, "Command: {}".format(ROAD_OPTIONS[state["command"]]), "l")
        vis_img = self.write_text(
            vis_img, "Waypoint dist: {:.2f}".format(state["waypoint_dist"]), "l")
        vis_img = self.write_text(
            vis_img, "Waypoint angle: {:.1f}".format(state["waypoint_angle"] * 180), "l")
        vis_img = self.write_text(
            vis_img, "Lane dist: {:.2f}".format(state["lane_dist"]), "l")
        vis_img = self.write_text(
            vis_img, "Lane angle: {:.1f}".format(state["lane_angle"] * 180), "l")
        if "tl_state" in state:
            vis_img = self.write_text(
                vis_img, "TL state: {}".format(state["tl_state"]), "l")
        if "tl_dist" in state:
            vis_img = self.write_text(
                vis_img, "TL dist: {:.2f}".format(state["tl_dist"]), "l")
        if "is_junction" in state:
            vis_img = self.write_text(
                vis_img, "Is junction: {}".format(state["is_junction"]), "l")
        vis_img = self.write_text(
            vis_img, "Is hazard: {}".format(state["hazard"]), "l")
        vis_img = self.write_text(
            vis_img, "Hazard dist: {:.2f}".format(state["hazard_dist"]), "l")

        vis_img = self.write_text(
            vis_img, "Speed: {:.2f}".format(state["speed"]), "r")
        vis_img = self.write_text(
            vis_img, "Speed Action: {:.2f}".format(avg_action[0]), "r")
        vis_img = self.write_text(
            vis_img, "Opt Speed: {:.2f}".format(state["optimal_speed"]), "r")
        vis_img = self.write_text(
            vis_img, "Steer Action: {:.2f}".format(avg_action[1]), "r")
        for reward, val in reward_dict.items():
            vis_img = self.write_text(
                vis_img, "{}: {:.2f}".format(reward, val), "r")

        cv2.imshow("Debug Display", vis_img)
        if state["hazard_coords"] is not None:
            self.draw_hazard_zones(state["hazard_coords"])
        cv2.waitKey(1)


def get_visualizer(config) -> BaseVisualizer:
    visualizer_type = config["visualizer_type"]
    if visualizer_type == "dummy":
        visualizer = DummyVisualizer(config)
    elif visualizer_type == "rl":
        visualizer = RLVisualizer(config)
    else:
        raise KeyError("visualizer type: {} is invalid".format(visualizer_type))
    return visualizer
