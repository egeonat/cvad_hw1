import random
from math import sqrt

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption
from utils.carla_utils import (angle_to_waypoint, draw_waypoints,
                               get_forward_speed)
from utils.geo import gnss_to_carla_coord


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # Lane changing
        if curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When road option changes
        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT,
                                                                RoadOption.CHANGELANERIGHT):
            if (i-1) not in ids_to_sample:
                ids_to_sample.append(max(0, i-1))
                dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].transform.location
            prev_location = route[i-1][0].transform.location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


class PlanManager():
    def __init__(self, dense_wp_interval, sparse_wp_interval, debug):
        self._dense_wp_int = dense_wp_interval
        self._sparse_wp_int = sparse_wp_interval
        self._debug = debug

        self._ego_vehicle = None
        self._world = None
        self.sparse_plan = None
        self.dense_plan = None
        self.sparse_wp_ind = 0
        self.dense_wp_ind = 0

    def reset(self, world, ego_vehicle, opendrive_map):
        self._ego_vehicle = ego_vehicle
        self._world = world
        vehicle_loc = ego_vehicle.get_transform().location
        vehicle_forward_vec = ego_vehicle.get_transform().get_forward_vector()
        starting_loc = vehicle_loc + vehicle_forward_vec * 2

        planner = GlobalRoutePlanner(GlobalRoutePlannerDAO(opendrive_map, self._dense_wp_int))
        planner.setup()

        spawn_points = opendrive_map.get_spawn_points()
        sparse_plan_length = 0
        # If the generated plan is too short, we reselect a destination
        while sparse_plan_length < 3:
            dest = random.choice(spawn_points).location

            self.dense_plan = planner.trace_route(starting_loc, dest)
            # Sparse plan is just a subsampling of the dense plan
            sparse_ids = downsample_route(self.dense_plan, self._sparse_wp_int)
            self.sparse_plan = [self.dense_plan[idx] for idx in sparse_ids]
            sparse_plan_length = len(self.sparse_plan)

        self.sparse_wp_ind = 0
        self.dense_wp_ind = 0

    def step(self, gps):
        speed = get_forward_speed(self._ego_vehicle)
        self.dense_wp_ind = self._update_plan(
            gps, speed, self.dense_plan, self.dense_wp_ind, False)
        self.sparse_wp_ind = self._update_plan(
            gps, speed, self.sparse_plan, self.sparse_wp_ind, True)
        dense_target = self.dense_plan[self.dense_wp_ind]
        sparse_target = self.sparse_plan[self.sparse_wp_ind]
        if self._debug:
            self._draw_targets(dense_target, sparse_target)
        return dense_target, sparse_target

    def prev_dense_target(self):
        prev_dense_command = None
        if self.dense_wp_ind > 0:
            prev_dense_command = self.dense_plan[self.dense_wp_ind - 1]
        return prev_dense_command

    def _update_plan(self, gps, speed, plan, wp_ind, look_ahead):
        gps_coords = gnss_to_carla_coord(gps.latitude, gps.longitude, gps.altitude)
        while wp_ind < len(plan) - 1:
            target_wp = plan[wp_ind][0]
            wp_loc = target_wp.transform.location
            wp_angle = angle_to_waypoint(self._ego_vehicle, target_wp)
            wp_dist = sqrt((gps_coords[0] - wp_loc.x) ** 2
                           + (gps_coords[1] - wp_loc.y) ** 2)

            needs_update = ((abs(wp_angle) > 0.5 and wp_dist < 10)
                            or (look_ahead and abs(wp_angle) < 0.25 and wp_dist < speed)
                            or wp_dist < 1.0)

            if needs_update:
                wp_ind += 1
            else:
                break
        return wp_ind

    def is_route_completed(self):
        is_route_completed = (len(self.sparse_plan) <= self.sparse_wp_ind + 1
                              or len(self.dense_plan) <= self.dense_wp_ind + 1)
        return is_route_completed

    def _draw_targets(self, dense_target, sparse_target):
        """If in debug mode, draw next waypoint on the CARLA world."""
        draw_waypoints(self._world, [sparse_target[0]], z=1.0)
        draw_waypoints(self._world, [dense_target[0]], color=(0, 0, 255))
