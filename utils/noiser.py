import numpy as np


class Noiser():
    def __init__(self, fps=20):
        self.noising = False
        self.fps = fps
        self.noise_duration = 0
        self.noise_timer = 0
        self.max_noise = 0

    def tick(self, steer):
        if not self.noising:
            if np.random.rand() < 0.001:
                print("Started noising!")
                self.noising = True
                # Max noise is uniformly random between 0.2 and 0.3
                self.max_noise = np.random.rand() * 0.2 + 0.3
                # Random selection between positive and negative
                self.max_noise *= (-1) ** (np.random.rand() > 0.5)
                # Noise duration is set to be between 2 and 5 seconds
                self.noise_duration = int((np.random.rand() * 3.0 + 1.5) * self.fps)
                self.noise_timer = 0
        if self.noising:
            dist_to_peak = abs(self.noise_duration/2 - self.noise_timer) / self.noise_duration
            noise = self.max_noise * (1 - dist_to_peak*2)
            steer = np.clip(steer + noise, -1, 1)
            self.noise_timer += 1
            if self.noise_timer > self.noise_duration:
                self.noising = False
        return steer
