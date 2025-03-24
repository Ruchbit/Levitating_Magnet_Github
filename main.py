import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Electromagnet class
class Electromagnet:
    def __init__(self, max_force=10):
        self.force = 0
        self.max_force = max_force

    def update(self, control_signal):
        """Update magnet force based on control input."""
        self.force = np.clip(control_signal, -self.max_force, self.max_force)

# Object class
class Object:
    def __init__(self, mass=1):
        self.position = 5  # Initial position
        self.velocity = 0
        self.acceleration = 0
        self.mass = mass

    def apply_force(self, force, noise=0):
        """Apply force to the object and update its motion."""
        dt = 0.01  # Time step
        self.acceleration = (force + noise) / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

# Noise generator
def noise_generator(noise_level=0.1):
    return np.random.uniform(-noise_level, noise_level)

