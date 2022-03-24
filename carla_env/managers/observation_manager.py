from itertools import chain

import numpy as np
from carla_env.managers.collision_handler import Collision
from shapely.geometry import LineString, Polygon
from utils.carla_utils import (_numpy, angle_to_waypoint, distance_to_waypoint,
                               get_forward_speed, get_tl_features)


class ObservationManager():
    def __init__(self, features_list, speed_limit):
        self._features_list = features_list
        self._speed_limit = speed_limit

        self._collision_handler = None
        self._tl_handler = None
        self._world = None
        self._ego_vehicle = None

    def reset(self, world, ego_vehicle, opendrive_map):
        self._collision_handler = Collision(ego_vehicle, world)
        self._world = world
        self._ego_vehicle = ego_vehicle
        self._opendrive_map = opendrive_map

    def get_state(self, dense_target, sparse_target, prev_dense_target):
        state = {}
        state["collision"] = self._collision_handler.tick(self._ego_vehicle)
        state["speed"] = get_forward_speed(self._ego_vehicle)
        state["waypoint_dist"] = distance_to_waypoint(self._ego_vehicle, sparse_target[0])
        state["waypoint_angle"] = angle_to_waypoint(self._ego_vehicle, sparse_target[0])
        state["command"] = sparse_target[1].value - 1
        route_features = self._get_route_fts(prev_dense_target, dense_target)
        state["route_dist"], state["route_angle"] = route_features

        lane_features = self._get_lane_features()
        state["lane_dist"], state["lane_angle"], state["is_junction"] = lane_features

        actors = self._world.get_actors()
        vehicle_list = self._filter_nearby_actors(actors.filter("*vehicle*"))
        walker_list = self._filter_nearby_actors(actors.filter("*walker*"))
        tl_list = self._filter_nearby_actors(actors.filter("*traffic_light*"))

        hazard_features = self._get_hazard_features(
            vehicle_list, walker_list, state["speed"], debug=False)
        state["hazard"], state["hazard_dist"], state["hazard_coords"] = hazard_features

        state["tl_state"], state["tl_dist"] = get_tl_features(
            self._ego_vehicle, tl_list, self._opendrive_map)
        state["tl_dist"] = state["tl_dist"] - 5.0

        state["optimal_speed"] = self._get_optimal_speed(
            state["hazard_dist"], state["tl_dist"], state["tl_state"])

        measurements = [state["waypoint_dist"],
                        state["waypoint_angle"],
                        ]

        if "lane" in self._features_list:
            measurements.append(state["lane_dist"])
            measurements.append(state["lane_angle"])
            measurements.append(state["is_junction"])
        if "hazard" in self._features_list:
            measurements.append(state["hazard_dist"])
        if "tl" in self._features_list:
            measurements.append(state["tl_state"])
            measurements.append(state["tl_dist"])

        state["measurements"] = np.array(measurements)
        return state

    def _get_route_fts(self, prev_wp, target_wp):
        vehicle_loc_np = _numpy(self._ego_vehicle.get_location())
        # Unit vector pointing in vehicle's direction
        forward_vec_np = _numpy(
            self._ego_vehicle.get_transform().rotation.get_forward_vector())

        if prev_wp is None:
            wp0_loc = vehicle_loc_np
        else:
            wp0_loc = _numpy(prev_wp[0].transform.location)
        wp1_loc = _numpy(target_wp[0].transform.location)

        # Get the vector to target_wp in the coordinate system with prev_wp as origin
        route_vec = wp1_loc - wp0_loc

        # Move vehicle_loc_np to the coordinate system with prev_wp as origin
        rel_vehicle_loc = vehicle_loc_np - wp0_loc

        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        # This link shows the equation used to get the distance to the optimal line
        route_dist = (
            ((route_vec[0]) * (-rel_vehicle_loc[1]) - (route_vec[1]) * (-rel_vehicle_loc[0]))
            / (np.linalg.norm(route_vec) + 1e-6)
        )
        route_angle = np.arctan2(route_vec[0], route_vec[1])
        forward_vec_angle = np.arctan2(forward_vec_np[0], forward_vec_np[1])
        angle_deviation = forward_vec_angle - route_angle
        if angle_deviation > np.pi:
            angle_deviation -= 2 * np.pi
        elif angle_deviation < - np.pi:
            angle_deviation += 2 * np.pi
        route_angle = angle_deviation / np.pi
        return route_dist, route_angle

    def _get_lane_features(self):
        ego_transform = self._ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_yaw = ego_transform.rotation.yaw

        lane_wp = self._opendrive_map.get_waypoint(ego_loc)
        lane_loc = lane_wp.transform.location
        lane_yaw = lane_wp.transform.rotation.yaw

        lane_dist = lane_loc.distance(ego_loc)
        lane_angle = ego_yaw - lane_yaw

        if lane_angle > 180:
            lane_angle -= 360
        elif lane_angle < - 180:
            lane_angle += 360
        lane_angle = lane_angle / 180

        is_junction = lane_wp.is_junction

        return lane_dist, lane_angle, is_junction

    def _filter_nearby_actors(self, actor_list, dist_limit=30):
        ego_loc = _numpy(self._ego_vehicle.get_transform().location)
        filtered_list = []
        for a in actor_list:
            if a.id != self._ego_vehicle.id:
                a_transform = a.get_transform()
                a_loc = _numpy(a_transform.location)
                dist = np.linalg.norm(ego_loc - a_loc)
                if dist <= dist_limit:
                    filtered_list.append(a)
        return filtered_list

    def _get_hazard_features(self, vehicles, walkers, speed, debug=False):
        ego_transform = self._ego_vehicle.get_transform()
        ego_loc = _numpy(ego_transform.location)
        ego_forward = _numpy(ego_transform.get_forward_vector(), normalize=True)
        ego_right = _numpy(ego_transform.get_right_vector(), normalize=True)
        ego_length = self._ego_vehicle.bounding_box.extent.x
        ego_width = self._ego_vehicle.bounding_box.extent.y

        vehicle_hazard_zones = self._get_vehicle_hazard_zones(ego_loc, ego_forward, vehicles)
        walker_hazard_zones = self._get_walker_hazard_zones(ego_loc, walkers)
        ego_zone = Polygon([
            ego_loc - ego_right * ego_width + ego_forward * (5.0 + ego_length + speed * 1.8),
            ego_loc + ego_right * ego_width + ego_forward * (5.0 + ego_length + speed * 1.8),
            ego_loc + ego_right * ego_width + ego_forward * ego_length,
            ego_loc - ego_right * ego_width + ego_forward * ego_length,
        ])
        ego_line = LineString([
            ego_loc - ego_right * ego_width + ego_forward * ego_length,
            ego_loc + ego_right * ego_width + ego_forward * ego_length,
        ])

        hazard_dist = 25
        is_hazard = False
        for p in chain(vehicle_hazard_zones, walker_hazard_zones):
            if p.intersects(ego_zone):
                dist = ego_line.distance(p)
                hazard_dist = min(hazard_dist, dist)
                is_hazard = True

        coords = None
        if debug:
            v_coords = [p.exterior.xy for p in vehicle_hazard_zones]
            w_coords = [p.exterior.xy for p in walker_hazard_zones]
            ego_coords = ego_zone.exterior.xy
            coords = (v_coords, w_coords, ego_coords)
        return is_hazard, hazard_dist, coords

    def _get_vehicle_hazard_zones(self, ego_loc, ego_forward, vehicles):
        hazard_zones = []
        for v in vehicles:
            if v.id == self._ego_vehicle.id:
                continue
            v_transform = v.get_transform()
            v_loc = _numpy(v_transform.location)
            v_forward = _numpy(v_transform.get_forward_vector(), normalize=True)
            heading_angle = np.arccos(np.dot(ego_forward, v_forward)) / np.pi * 180
            if heading_angle > 135 and heading_angle < 225:
                continue

            v_length = v.bounding_box.extent.x
            v_width = v.bounding_box.extent.y
            v_right = _numpy(v_transform.get_right_vector(), normalize=True)
            v_poly = Polygon([
                v_loc - v_right * (0.3 + v_width) + v_forward * (1.0 + v_length),
                v_loc + v_right * (0.3 + v_width) + v_forward * (1.0 + v_length),
                v_loc + v_right * (0.3 + v_width) - v_forward * (0.5 + v_length),
                v_loc - v_right * (0.3 + v_width) - v_forward * (0.5 + v_length),
            ])
            hazard_zones.append(v_poly)
        return hazard_zones

    def _get_walker_hazard_zones(self, ego_loc, walkers):
        hazard_zones = []
        for w in walkers:
            w_transform = w.get_transform()
            w_loc = _numpy(w_transform.location)
            w_forward = _numpy(w_transform.get_forward_vector(), normalize=True)
            w_right = _numpy(w_transform.get_right_vector(), normalize=True)
            w_poly = Polygon([
                w_loc - w_right * 0.6 + w_forward * 0.8,
                w_loc + w_right * 0.6 + w_forward * 0.8,
                w_loc + w_right * 0.6 - w_forward * 0.3,
                w_loc - w_right * 0.6 - w_forward * 0.3,
            ])
            hazard_zones.append(w_poly)
        return hazard_zones

    def _get_optimal_speed(self, hazard_dist, tl_dist, tl_state):
        min_dist = hazard_dist
        if tl_state == 1:
            min_dist = min(min_dist, tl_dist)
        min_dist = np.clip(min_dist, 2, 22)
        optimal_speed = ((min_dist - 2) / 20) * self._speed_limit
        return optimal_speed

    def cleanup(self):
        if self._collision_handler is not None:
            self._collision_handler.clean()
