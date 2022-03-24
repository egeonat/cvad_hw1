"""This module contains utility functions that deals with the CARLA API."""
import math

import carla
import numpy as np
from agents.tools.misc import is_within_distance


def distance_to_waypoint(actor, waypoint):
    """Return distance between actor and waypoint."""
    vehicle_loc = actor.get_location()
    waypoint_loc = waypoint.transform.location
    return vehicle_loc.distance(waypoint_loc)


def vec_towards_wp(actor, waypoint):
    """Return numpy array representing vector from actor to waypoint."""
    waypoint_loc = waypoint.transform.location
    waypoint_loc_np = _numpy(waypoint_loc)

    vehicle_loc = actor.get_location()
    vehicle_loc_np = _numpy(vehicle_loc)
    vector_to_waypoint = waypoint_loc_np - vehicle_loc_np
    return vector_to_waypoint


def angle_to_waypoint(actor, waypoint):
    """
    Return angle between the heading of the actor and the waypoint's direction in the range
    of [-1, 1].
    """
    forward_vec = actor.get_transform().rotation.get_forward_vector()
    forward_vec_np = _numpy(forward_vec, normalize=True)
    forward_vec_angle = np.arctan2(forward_vec_np[0], forward_vec_np[1])

    target_vec_np = vec_towards_wp(actor, waypoint)
    target_vec_np /= np.linalg.norm(target_vec_np)
    target_vec_angle = np.arctan2(target_vec_np[0], target_vec_np[1])

    angle_deviation = forward_vec_angle - target_vec_angle
    if angle_deviation > np.pi:
        angle_deviation = 2 * np.pi - angle_deviation
    elif angle_deviation < - np.pi:
        angle_deviation = 2 * np.pi + angle_deviation
    wp_angle = angle_deviation / np.pi
    return wp_angle


def relative_position_to_road_center(ego_vehicle, carla_map):
    """Return distance and yaw difference between vehicle and nearest road center."""
    vehicle_transform = ego_vehicle.get_transform()
    road_transform = carla_map.get_waypoint(vehicle_transform.location).transform
    distance = road_transform.location.distance(vehicle_transform.location)
    angle_dif = road_transform.rotation.yaw - vehicle_transform.rotation.yaw
    if angle_dif < -180:
        angle_dif += 360
    if angle_dif > 180:
        angle_dif -= 360
    # Change range to -1, 1 where 0 faces forward
    angle_dif /= 180
    return distance, angle_dif


def get_forward_speed(ego_vehicle):
    """Convert the vehicle transform directly to forward speed """
    velocity = ego_vehicle.get_velocity()
    transform = ego_vehicle.get_transform()

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw),
                            np.cos(pitch) * np.sin(yaw),
                            np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


def _is_light_red(vehicle):
    if vehicle.get_traffic_light_state() == carla.libcarla.TrafficLightState.Green:
        return None
    return True


def _numpy(carla_vector, *, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-6)

    return result


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def is_walker_hazard(world, ego_vehicle, walkers_list):
    p1 = _numpy(ego_vehicle.get_location())
    v1 = 10.0 * _orientation(ego_vehicle.get_transform().rotation.yaw)

    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat

        collides, collision_point = get_collision(p1, v1, p2, v2)
        if collides:
            return walker

    return None


def blocking_walker_dist(world, ego_vehicle, walkers_list):
    p1 = _numpy(ego_vehicle.get_location())
    v1 = 10.0 * _orientation(ego_vehicle.get_transform().rotation.yaw)

    closest_dist = float("inf")
    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))

        if s2 < 0.05:
            v2_hat *= s2

        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat

        collides, collision_point = get_collision(p1, v1, p2, v2)

        if collides:
            dist = np.linalg.norm(p1 - collision_point)
            closest_dist = min(dist, closest_dist)

    if closest_dist == float("inf"):
        return -1
    return closest_dist


def is_vehicle_hazard(world, ego_vehicle, vehicle_list):
    o1 = _orientation(ego_vehicle.get_transform().rotation.yaw)
    p1 = _numpy(ego_vehicle.get_location())
    s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(ego_vehicle.get_velocity())))
    v1_hat = o1

    for target_actor in vehicle_list:
        if target_actor.id == ego_vehicle.id:
            continue

        o2 = _orientation(target_actor.get_transform().rotation.yaw)
        p2 = _numpy(target_actor.get_location())

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        if angle_between_heading > 60.0 and not (angle_to_car < 30 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue
        return target_actor

    return None


def blocking_vehicle_dist(world, ego_vehicle, vehicle_list, default_val=100):
    o1 = _orientation(ego_vehicle.get_transform().rotation.yaw)
    p1 = _numpy(ego_vehicle.get_location())
    s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(ego_vehicle.get_velocity())))
    v1_hat = o1

    closest_dist = default_val
    for target_actor in vehicle_list:
        if target_actor.id == ego_vehicle.id:
            continue

        o2 = _orientation(target_actor.get_transform().rotation.yaw)
        p2 = _numpy(target_actor.get_location())

        p2_p1 = p2 - p1
        distance = np.linalg.norm(p2_p1)
        p2_p1_hat = p2_p1 / (distance + 1e-4)

        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1:
            continue
        else:
            closest_dist = min(distance, closest_dist)
    return closest_dist


def draw_waypoints(world, waypoints, z=0.5, color=(255, 0, 0)):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        angle = math.radians(wpt_t.rotation.yaw)
        begin = wpt_t.location + carla.Location(z=z)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        color = carla.Color(*color, 255)
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0, color=color)


def get_tl_features(vehicle, tl_list, carla_map):
    # 0 represents green or nonexistent light. 1 represents yellow or red light
    tl_state = 0
    # 50 if no light detected
    tl_dist = 50

    vehicle_loc = vehicle.get_location()
    vehicle_yaw = vehicle.get_transform().rotation.yaw
    vehicle_wp = carla_map.get_waypoint(vehicle_loc)
    for tl in tl_list:
        tl_transform = tl.get_transform()
        tl_loc = tl_transform.location
        tl_waypoint = carla_map.get_waypoint(tl_loc)

        if tl_waypoint.road_id != vehicle_wp.road_id:
            continue

        if is_within_distance(tl_loc, vehicle_loc, vehicle_yaw, 25, 120):
            tl_dist = np.linalg.norm(_numpy(vehicle_loc) - _numpy(tl_loc))
            tl_state = int(vehicle.get_traffic_light_state())
            tl_state = int(tl_state == 0 or tl_state == 1)

    return tl_state, tl_dist
