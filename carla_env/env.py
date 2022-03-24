import random
from time import sleep

import carla
from func_timeout import FunctionTimedOut, func_timeout
from utils.carla_server import kill_carla_server, start_carla_server

from carla_env.managers.actor_manager import ActorManager
from carla_env.managers.observation_manager import ObservationManager
from carla_env.managers.plan_manager import PlanManager
from carla_env.managers.reward_manager import RewardManager
from carla_env.managers.sensor_manager import SensorManager
from carla_env.managers.weather_manager import WeatherManager


class Env():
    def __init__(self, config=None):
        # Variables that do not change between episodes
        self.fps = config["fps"]
        self.random_maps = config["random_maps"]
        self.route_dist_limit = config["route_dist_limit"]

        # Start server
        start_carla_server(config["server_port"])
        # Create client and traffic manager instances
        self.client = None
        self.traffic_manager = None
        self.make_client(config["server_port"])
        self.make_tm()

        blp_lib = self._world.get_blueprint_library()
        # Create managers for various aspects of the environment
        self._sensor_manager = SensorManager(blp_lib, config["sensors"])
        self._reward_manager = RewardManager()
        self._actor_manager = ActorManager(
            self.client, blp_lib, config["num_walkers"], config["num_vehicles"],
            config["ego_spawn_point_idx"])
        self._weather_manager = WeatherManager(config["dynamic_weather"], 1 / self.fps)
        self._plan_manager = PlanManager(
            config["dense_wp_interval"], config["sparse_wp_interval"], config["debug"])
        self._obs_manager = ObservationManager(config["features"], config["speed_limit"])

        # Counter for episodes
        self.episode_counter = 0

        # Variables that change for every episode
        self._world = self.client.load_world(config["map"])
        self._opendrive_map = self._world.get_map()
        self._ego_vehicle = None

        # Set synchronous mode for client and tm
        self._set_synchronous_mode(True)

        # Variables that change during episodes
        self.current_step = -1

    def make_client(self, server_port):
        """Create client and world for the environment. Called in __init__"""
        client_is_initialized = False
        # This sleep is to wait until the carla server is up and running.
        # Otherwise we print an error
        sleep(4)
        print("Creating client")
        counter = 0
        while not client_is_initialized:
            try:
                counter += 1
                self.client = carla.Client("localhost", server_port)
                self.client.set_timeout(20.0)
                self._world = self.client.get_world()
                client_is_initialized = True
            except RuntimeError as err:
                if counter > 3:
                    print(err)
                    print("Trying again...")

    def make_tm(self):
        print("Creating tm")
        tm_port = 9500
        tm_is_initialized = False
        while not tm_is_initialized:
            try:
                self.traffic_manager = self.client.get_trafficmanager(tm_port)
                tm_is_initialized = True
            except Exception as err:
                print("Caught exception during traffic manager creation: ")
                print(err)
                tm_port += 1
                print("Trying with port {}...".format(tm_port))

    def reset(self):
        """Resets the environment."""
        self.episode_counter += 1

        self._cleanup()
        if self.random_maps:
            new_map = random.choice(self.client.get_available_maps())
            self._world = self.client.load_world(new_map)
            self._opendrive_map = self._world.get_map()
            self._set_synchronous_mode(True)

        self._actor_manager.reset(self._world, self._opendrive_map)
        self._ego_vehicle = self._actor_manager.spawn_ego_vehicle()
        self._sensor_manager.reset(self._world, self._ego_vehicle)
        self._weather_manager.reset(self._world)
        self._plan_manager.reset(self._world, self._ego_vehicle, self._opendrive_map)
        self._obs_manager.reset(self._world, self._ego_vehicle, self._opendrive_map)

        self.current_step = -1

        self._actor_manager.spawn_vehicles()
        self._actor_manager.spawn_walkers()
        self._sensor_manager.spawn_sensors(self._world)

        self._move_spectator()

        # Commands may not register in the first few seconds, so we skip them
        for i in range(self.fps * 2):
            current_frame = self._world.tick()

        sensor_data = self._sensor_manager.tick(current_frame)
        dense_target, sparse_target = self._plan_manager.step(sensor_data["gps"])
        prev_dense_target = self._plan_manager.prev_dense_target()
        state = self._obs_manager.get_state(dense_target, sparse_target, prev_dense_target)
        state.update(sensor_data)

        fake_action = {
            "throttle": 0,
            "brake": 0,
            "steer": 0
        }
        reward_dict = self._reward_manager.get_reward(state, fake_action)
        is_terminal = self._get_terminal(state)
        return state, reward_dict, is_terminal

    def step(self, action):
        self.current_step += 1
        control = carla.VehicleControl(
            throttle=action["throttle"],
            brake=action["brake"],
            steer=action["steer"]
        )
        self._ego_vehicle.apply_control(control)
        self._move_spectator()

        current_frame = self._world.tick()
        self._weather_manager.tick()
        self._actor_manager.update_lights(self._weather_manager.weather)

        sensor_data = self._sensor_manager.tick(current_frame)
        dense_target, sparse_target = self._plan_manager.step(sensor_data["gps"])
        prev_dense_target = self._plan_manager.prev_dense_target()
        state = self._obs_manager.get_state(dense_target, sparse_target, prev_dense_target)
        state.update(sensor_data)

        reward_dict = self._reward_manager.get_reward(state, action)
        is_terminal = self._get_terminal(state)
        return state, reward_dict, is_terminal

    def _get_terminal(self, state):
        # is_terminal is either empty, or contains our cause for termination as a str
        is_terminal = []
        if state["collision"]:
            is_terminal.append("collision")
            print("Collision occured.", " " * 40)
        if self._plan_manager.is_route_completed():
            is_terminal.append("finished")
            print("Reached last waypoint.", " " * 40)
        if abs(state["route_dist"]) > self.route_dist_limit:
            is_terminal.append("route_dist")
            print("Got too far from lane center.", " " * 40)

        return is_terminal

    def _set_synchronous_mode(self, sync):
        """Set or unset synchronous mode for the server and the traffic manager."""
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        if sync:
            settings.fixed_delta_seconds = 1 / self.fps
        else:
            settings.fixed_delta_seconds = None
        self._world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(sync)

    def _move_spectator(self):
        """Move simulator camera to vehicle for viewing."""
        spectator = self._world.get_spectator()
        transform = self._ego_vehicle.get_transform()
        transform.location.z += 20
        transform.rotation.pitch = -90
        transform.rotation.roll = 0
        transform.rotation.yaw = 0
        spectator.set_transform(transform)

    def _cleanup(self):
        """Destroy leftover actors."""
        self._sensor_manager.cleanup()
        self._actor_manager.cleanup()
        self._obs_manager.cleanup()
        self._world.tick()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            print("Exiting...")
            func_timeout(10, self._cleanup)
            func_timeout(10, self._set_synchronous_mode, (False,))
            kill_carla_server()
        except FunctionTimedOut:
            print("Timeout while attempting to set CARLA to async mode.")
        except Exception as err:
            print(err)
