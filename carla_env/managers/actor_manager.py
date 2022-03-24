import random

import carla
from carla import VehicleLightState


class ActorManager():
    def __init__(self, client, blp_lib, num_walkers, num_vehicles, ego_spawn_point_idx):
        # Fixed attributes
        self._client = client
        self._blp_lib = blp_lib
        self._num_walkers = num_walkers
        self._num_vehicles = num_vehicles
        self._ego_spawn_point_idx = ego_spawn_point_idx

        # Variables that change on every reset
        self._world = None
        self._spawn_points = None
        self._light_state = None
        self._ego_vehicle = None
        self._vehicles = []
        self._walkers = []
        self._walker_controllers = []

    def reset(self, world, opendrive_map):
        """Reset world and spawn_points on episode reset."""
        self._world = world
        self._spawn_points = opendrive_map.get_spawn_points()
        self._light_state = VehicleLightState.Position
        self._ego_vehicle = None
        self._vehicles = []
        self._walkers = []
        self._walker_controllers = []

    def spawn_ego_vehicle(self):
        """Spawn and return ego vehicle."""
        ego_bp = self._blp_lib.find("vehicle.lincoln.mkz2017")
        if self._ego_spawn_point_idx is None:
            spawn_point_idx = random.randint(0, len(self._spawn_points) - 1)
        else:
            spawn_point_idx = self._ego_spawn_point_idx
        ego_transform = self._spawn_points.pop(spawn_point_idx)
        self._ego_vehicle = self._world.spawn_actor(ego_bp, ego_transform)
        # Do one tick so the ego vehicle is spawned on the server
        self._world.tick()
        return self._ego_vehicle

    def spawn_walkers(self, running_walker_ratio=0.4, crossing_factor=0.9):
        """Generate walkers."""
        SpawnActor = carla.command.SpawnActor

        # Generate spawn points
        spawn_points = []
        for i in range(self._num_walkers):
            spawn_point = carla.Transform()
            loc = self._world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # Spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(self._blp_lib.filter("*walker*"))
            # Set as not invincible
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            # Set the max speed
            if walker_bp.has_attribute("speed"):
                if random.random() > running_walker_ratio:
                    # walking
                    walker_speed.append(walker_bp.get_attribute("speed").recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute("speed").recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        walker_speed2 = []
        num_failures = 0
        for i, response in enumerate(self._client.apply_batch_sync(batch, True)):
            if response.error:
                num_failures += 1
            else:
                self._walkers.append(response.actor_id)
                walker_speed2.append(walker_speed[i])

        if num_failures > 0:
            print("Couldn't spawn {} walkers because of collisions.".format(num_failures))
        walker_speed = walker_speed2

        self._world.tick()

        # Spawn the walker controller
        batch = []
        walker_controller_bp = self._world.get_blueprint_library().find("controller.ai.walker")
        for i, walker_idx in enumerate(self._walkers):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walker_idx))
        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                print("Error while spawning walker controller: ", response.error)
            else:
                self._walker_controllers.append(response.actor_id)

        self._world.set_pedestrians_cross_factor(crossing_factor)
        for walker_controller in self._world.get_actors(self._walker_controllers):
            # start walker
            walker_controller.start()
            # set walk to random point
            walker_controller.go_to_location(self._world.get_random_location_from_navigation())
            # max speed
            walker_controller.set_max_speed(float(walker_speed[i]))

    def spawn_vehicles(self):
        """Generate non-ego vehicles."""
        random.shuffle(self._spawn_points)
        num_spawn_points = len(self._spawn_points)
        num_vehicles = self._num_vehicles
        if num_vehicles > num_spawn_points:
            print("Couldn't spawn {} vehicles as only {} spawn points available".format(
                num_vehicles, num_spawn_points))
            num_vehicles = num_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        blueprints = self._blp_lib.filter("*vehicle*")
        batch = []
        for n, transform in enumerate(self._spawn_points):
            if n >= num_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(
                SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True))
            )
            self._spawn_points.pop(0)

        self._vehicles = []
        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                print("Error while spawning vehicle: ", response.error)
            else:
                self._vehicles.append(
                    (response.actor_id, self._world.get_actor(response.actor_id)))

    def update_lights(self, weather):
        light_state = VehicleLightState.Position
        # Turn on vehicle lights if it is night
        if weather.sun_altitude_angle < 0:
            light_state |= VehicleLightState.LowBeam
        # Turn on fog lights
        if weather.fog_density > 20:
            light_state |= VehicleLightState.Fog

        # Modify vehicle light state if a change is necessary
        if light_state != self._light_state:
            for vehicle_idx, vehicle in self._vehicles:
                vehicle.set_light_state(VehicleLightState(light_state))
            self._ego_vehicle.set_light_state(VehicleLightState(light_state))
            self._light_state = light_state

    def cleanup(self):
        if self._world is not None:
            self._client.apply_batch(
                [carla.command.DestroyActor(x[0]) for x in self._vehicles])
            self._vehicles.clear()

            if self._ego_vehicle is not None:
                self._ego_vehicle.destroy()
            self._ego_vehicle = None

            for walker_controller in self._world.get_actors(self._walker_controllers):
                if walker_controller is not None:
                    walker_controller.stop()
                    walker_controller.destroy()
            self._walker_controllers.clear()

            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._walkers])
            self._walkers.clear()
