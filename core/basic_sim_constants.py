"""
This file contains constants for the new basic physics simulations.
"""

import numpy as np

# Constants for the simple harmonic oscillator
HARMONIC_OSCILLATOR = {
    "mass": 1.0,  # kg
    "k": 2.0,  # N/m
    "initial_position": 1.0,  # m
    "initial_velocity": 0.0,  # m/s
}

# Constants for the particle in a box
PARTICLE_IN_A_BOX = {
    "length": 1e-9,  # meters (e.g., a nanostructure)
    "mass": 9.10938356e-31,  # kg (electron mass)
}

# Constants for the RLC circuit
RLC_CIRCUIT = {
    "resistance": 10.0,  # ohms
    "inductance": 1e-3,  # henries (1 mH)
    "capacitance": 1e-6,  # farads (1 uF)
    "initial_charge": 0.01,  # coulombs
    "initial_current": 0.0,  # amperes
}
