"""
Defines the SensorManager class used by the CARLA env wrapper.
the suite itself is written to be extensible if necessary.
"""
from collections import deque
from time import sleep

import carla
from utils.data_utils import parse_carla_image, preprocess_semantic
from utils.mappings import LSS_SENSORS


def sensor_callback(name, queue, process_fn=None):
    """Generate sensor callback fn that appends to a queue and optionally processes data."""
    def cb(data):
        if process_fn is not None:
            data = process_fn(data)
        queue.append(data)

    return cb


class SensorManager():
    """Sensor suite for CARLA RL environment."""
    def __init__(self, blp_lib, requested_sensors):
        # Fixed attributes
        self._blp_lib = blp_lib
        # List of strings containing requested sensor names
        self._requested_sensors = requested_sensors

        # Variables that change with episodes
        self._world = None
        self._ego_vehicle = None
        # Map of str -> (sensor, queue) that holds active sensors
        self._active_sensors = {}

    def reset(self, world, ego_vehicle):
        self._world = world
        self._ego_vehicle = ego_vehicle

    def spawn_sensors(self, world):
        """Spawns sensors from self._requested_sensors"""
        for sensor_name in self._requested_sensors:
            q = deque()
            if sensor_name == "rgb":
                bp = self._create_camera_bp(sensor_name, 512, 120)
                transform = carla.Transform(
                    carla.Location(x=1.3, z=1.3))
                sensor = world.spawn_actor(bp, transform, attach_to=self._ego_vehicle)
                sensor.listen(sensor_callback(sensor_name, q))
                sensor_name = "rgb"
            elif "cam" in sensor_name:
                sdict = None
                for d in LSS_SENSORS:
                    if d["id"] == sensor_name:
                        sdict = d
                bp = self._create_camera_bp("rgb", sdict["width"], sdict["fov"])
                transform = carla.Transform(
                    carla.Location(x=sdict["x"], y=sdict["y"], z=sdict["z"]),
                    carla.Rotation(yaw=sdict["yaw"])
                )
                sensor = world.spawn_actor(bp, transform, attach_to=self._ego_vehicle)
                sensor.listen(sensor_callback(sensor_name, q))
            elif sensor_name == "semantic_bev":
                bp = self._create_camera_bp("semantic_segmentation", 256, 90)
                transform = carla.Transform(carla.Location(z=16), carla.Rotation(pitch=-90))
                sensor = world.spawn_actor(bp, transform, attach_to=self._ego_vehicle)
                sensor.listen(sensor_callback(sensor_name, q))
                sensor_name = "semantic_bev"
            elif sensor_name == "gps":
                bp = self._blp_lib.find("sensor.other.gnss")
                bp.set_attribute("noise_alt_stddev", str(0.0000005))
                bp.set_attribute("noise_lat_stddev", str(0.0000005))
                bp.set_attribute("noise_lon_stddev", str(0.0000005))
                sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._ego_vehicle)
                sensor.listen(sensor_callback(sensor_name, q))
            elif sensor_name == "imu":
                bp = self._blp_lib.find("sensor.other.imu")
                bp.set_attribute("noise_accel_stddev_x", str(0.001))
                bp.set_attribute("noise_accel_stddev_y", str(0.001))
                bp.set_attribute("noise_accel_stddev_z", str(0.015))
                bp.set_attribute("noise_gyro_stddev_x", str(0.001))
                bp.set_attribute("noise_gyro_stddev_y", str(0.001))
                bp.set_attribute("noise_gyro_stddev_z", str(0.001))
                sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._ego_vehicle)
                sensor.listen(sensor_callback(sensor_name, q))
            else:
                print("Sensor {} not found!".format(sensor_name))
            self._active_sensors[sensor_name] = (sensor, q)

    def tick(self, current_frame):
        """Get data from all sensors and return as a dict."""
        state = {}
        for sensor_name, (sensor, sensor_q) in self._active_sensors.items():
            state[sensor_name] = self._get_sensor_data(sensor_name, sensor_q, current_frame)
            if "semantic" in sensor_name:
                state[sensor_name] = preprocess_semantic(state[sensor_name])
            if "rgb" in sensor_name or "cam" in sensor_name:
                state[sensor_name] = parse_carla_image(state[sensor_name])
        return state

    def get_camera(self):
        return self._active_sensors["rgb"][0]

    def _get_sensor_data(self, sensor_name, sensor_queue, frame):
        """Read sensor data from its queue."""
        attempt = 0
        while True:
            attempt += 1
            try:
                data = sensor_queue.popleft()
                while data.frame < frame:
                    data = sensor_queue.popleft()
                return data
            except IndexError:
                err_str = "Index error attempting to read {}\t-\tattempt: {}".format(
                    sensor_name, attempt)
                if attempt == 5:
                    err_str = "\n" + err_str
                if attempt > 5:
                    print(err_str, end="\r")
                sleep(0.05)

    def _create_camera_bp(self, cam_type, resolution, fov):
        cam_bp = self._blp_lib.find("sensor.camera.{}".format(cam_type))
        cam_bp.set_attribute("image_size_x", str(resolution))
        cam_bp.set_attribute("image_size_y", str(resolution))
        cam_bp.set_attribute("fov", str(fov))
        if "semantic" not in cam_type:
            cam_bp.set_attribute("lens_circle_multiplier", str(3.0))
            cam_bp.set_attribute("lens_circle_falloff", str(3.0))
            cam_bp.set_attribute("chromatic_aberration_intensity", str(0.5))
            cam_bp.set_attribute("chromatic_aberration_offset", str(0))
        return cam_bp

    def _create_semantic_lidar_bp(self):
        lidar_bp = self._blp_lib.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("range", str(80))
        lidar_bp.set_attribute("rotation_frequency", str(20))
        lidar_bp.set_attribute("channels", str(64))
        lidar_bp.set_attribute("upper_fov", str(20))
        lidar_bp.set_attribute("lower_fov", str(-40))
        lidar_bp.set_attribute("points_per_second", str(1120000))
        return lidar_bp

    def cleanup(self):
        self._ego_vehicle = None
        for sensor_name, (sensor, _) in self._active_sensors.items():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._active_sensors.clear()
