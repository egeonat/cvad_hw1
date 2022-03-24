import carla

# Carla command values to strings
ROAD_OPTIONS = {
    0: "LEFT",
    1: "RIGHT",
    2: "STRAIGHT",
    3: "LANEFOLLOW",
    4: "CHANGELANELEFT",
    5: "CHANGELANERIGHT"
}

# Traffic light states are enums of the following order:
TRAFFIC_LIGHT_STATES = {
    carla.libcarla.TrafficLightState.Red: 0,
    carla.libcarla.TrafficLightState.Yellow: 1,
    carla.libcarla.TrafficLightState.Green: 2,
    carla.libcarla.TrafficLightState.Off: 3,
    carla.libcarla.TrafficLightState.Unknown: 4,
}

# Class label mappings
LABEL_MAPPING = {
    # Background classes
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    5: 0,
    9: 0,
    11: 0,
    12: 0,  # Should we add a new class for stop signs?
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    19: 0,
    21: 0,
    22: 0,
    # Road Markings
    6: 1,
    # Road
    7: 2,
    # Sidewalk
    8: 3,
    # Moving obstacles
    4: 4,
    10: 4,
    20: 4,
    # Traffic Lights
    18: 0
}

LSS_SENSORS = [
    {
        "type": "sensor.camera.rgb",
        "x": 1.6,
        "y": -0.3,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "width": 288,
        "height": 288,
        "fov": 100,
        "id": "front_left_cam"
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.6,
        "y": 0.3,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "width": 288,
        "height": 288,
        "fov": 100,
        "id": "front_right_cam"
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": -0.4,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -75.0,
        "width": 288,
        "height": 288,
        "fov": 100,
        "id": "side_left_cam"
    },
    {
        "type": "sensor.camera.rgb",
        "x": 1.3,
        "y": 0.4,
        "z": 1.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 75.0,
        "width": 288,
        "height": 288,
        "fov": 100,
        "id": "side_right_cam"
    }
]


LSS_DATA_AUG_CONF = {
    "final_dim": (288, 288),
    "H": 288, "W": 288,
    "resize_lim": (1.0, 1.05),
    "rot_lim": (-4.0, 4.0),
    "rand_flip": False,
    "bot_pct_lim": (0.0, 0.0),
    "aug": True,
    "normalize": True,
    "ncams": 4
}

# Frustum grid step sizes and step magnitudes for LSS (?)
LSS_GRID_CONF = {
    "xbound": [0, 16, 0.125],
    "ybound": [-16, 16, 0.125],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 15.0, 1.0]
}
