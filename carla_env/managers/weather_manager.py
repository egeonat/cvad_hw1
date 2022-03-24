#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Dynamic Weather:

Connect to a CARLA Simulator instance and control the weather. Change Sun
position smoothly with time and generate storms occasionally.
"""

import numpy as np


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self):
        self._t = np.random.rand() * 2.0 * np.pi
        self.azimuth = np.random.rand() * 360.0
        self.altitude = 90 * np.sin(self._t)

    def tick(self, delta_seconds):
        self._t = (self._t + 0.05 * delta_seconds) % (2.0 * np.pi)
        self.azimuth = (self.azimuth + 5.0 * delta_seconds) % 360.0
        self.altitude = 90 * np.sin(self._t)

    def __str__(self):
        return "Sun(alt: %.2f, azm: %.2f)" % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self):
        self._t = np.random.randint(-250.0, 100)
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = np.clip(delta + self._t, -250.0, 100.0)
        self.clouds = np.clip(self._t + 40.0, 0.0, 90.0)
        self.rain = np.clip(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = np.clip(self._t + delay, 0.0, 85.0)
        self.wetness = np.clip(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = np.clip(self._t - 10, 0.0, 40.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return "Storm(clouds=%d%%, rain=%d%%, wind=%d%%)" % (self.clouds, self.rain, self.wind)


class WeatherManager(object):
    def __init__(self, dynamic_weather, delta_seconds):
        self._dynamic_weather = dynamic_weather
        self._delta_seconds = delta_seconds

    def reset(self, world):
        self.weather = world.get_weather()

        self._world = world
        if self._dynamic_weather:
            self._sun = Sun()
            self._storm = Storm()

    def tick(self):
        if self._dynamic_weather:
            self._sun.tick(self._delta_seconds)
            self._storm.tick(self._delta_seconds)
            self.weather.cloudiness = self._storm.clouds
            self.weather.precipitation = self._storm.rain
            self.weather.precipitation_deposits = self._storm.puddles
            self.weather.wind_intensity = self._storm.wind
            self.weather.fog_density = self._storm.fog
            self.weather.wetness = self._storm.wetness
            self.weather.sun_azimuth_angle = self._sun.azimuth
            self.weather.sun_altitude_angle = self._sun.altitude
            self._world.set_weather(self.weather)
        return self.weather

    def __str__(self):
        return "%s %s" % (self._sun, self._storm)
