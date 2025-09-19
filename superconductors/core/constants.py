"""
Physical constants and conversion factors for strange metal simulations.

All constants are in SI units unless otherwise specified.
"""

import numpy as np
from typing import Dict, Optional


# Fundamental physical constants
PHYSICAL_CONSTANTS = {
    # Fundamental constants
    "hbar": 1.054571817e-34,  # J⋅s (reduced Planck constant)
    "kb": 1.380649e-23,  # J/K (Boltzmann constant)
    "e": 1.602176634e-19,  # C (elementary charge)
    "me": 9.1093837015e-31,  # kg (electron mass)
    "c": 299792458,  # m/s (speed of light)
    "mu_0": 1.25663706212e-6,  # N/A² (vacuum permeability)
    "epsilon_0": 8.8541878128e-12,  # F/m (vacuum permittivity)
    "mu_B": 9.2740100783e-24,  # J/T (Bohr magneton)
    "a_0": 5.29177210903e-11,  # m (Bohr radius)
    
    # Derived constants
    "alpha": 1/137.035999084,  # Fine structure constant
    "R_K": 25812.80745,  # Ω (von Klitzing constant)
    "G_0": 7.748091729e-5,  # S (conductance quantum, 2e²/h)
    "Phi_0": 2.067833848e-15,  # Wb (flux quantum, h/2e)
    
    # Energy conversions
    "eV_to_J": 1.602176634e-19,  # eV to Joules
    "meV_to_J": 1.602176634e-22,  # meV to Joules
    "K_to_eV": 8.617333262e-5,  # Kelvin to eV (kb/e)
    "THz_to_meV": 4.13566769,  # THz to meV
    
    # Length conversions
    "angstrom_to_m": 1e-10,  # Angstrom to meters
    "nm_to_m": 1e-9,  # nanometer to meters
    "um_to_m": 1e-6,  # micrometer to meters
}


# Material-specific constants for superconductors
MATERIAL_CONSTANTS = {
    # Typical values for cuprates
    "cuprate": {
        "lattice_constant": 3.8e-10,  # m (~3.8 Angstroms)
        "coherence_length": 1.5e-9,  # m (~15 Angstroms)
        "penetration_depth": 1.5e-7,  # m (~150 nm)
        "fermi_velocity": 2e5,  # m/s
        "plasma_frequency": 1e15,  # rad/s
        "debye_temperature": 400,  # K
        "gap_max": 30e-3,  # eV (max gap ~30 meV)
    },
    
    # Typical values for iron-based superconductors
    "iron_based": {
        "lattice_constant": 4.0e-10,  # m
        "coherence_length": 2e-9,  # m
        "penetration_depth": 2e-7,  # m
        "fermi_velocity": 1e5,  # m/s
        "plasma_frequency": 5e14,  # rad/s
        "debye_temperature": 300,  # K
        "gap_max": 15e-3,  # eV
    },
    
    # Planckian dissipation limit
    "planckian": {
        "scattering_rate_coefficient": 1.0,  # ℏ/τ = α⋅kb⋅T
        "resistivity_coefficient": 1.2,  # μΩ⋅cm/K for strange metals
    },
}


# Quantum simulation parameters (Pasqal specific)
SIMULATION_CONSTANTS = {
    # Rydberg atom parameters
    "rydberg": {
        "C6": 2.5e-24,  # J⋅m^6 (van der Waals coefficient)
        "Omega_max": 2 * np.pi * 10e6,  # rad/s (max Rabi frequency)
        "Delta_max": 2 * np.pi * 20e6,  # rad/s (max detuning)
        "lifetime": 100e-6,  # s (Rydberg state lifetime)
        "blockade_radius": 10e-6,  # m (~10 μm)
    },
    
    # Device constraints
    "pasqal": {
        "min_spacing": 4e-6,  # m (minimum atom spacing)
        "max_atoms": 256,  # Maximum number of atoms
        "position_precision": 0.1e-6,  # m (positioning accuracy)
        "shot_time": 1e-3,  # s (time per shot)
    },
    
    # Digital quantum gates
    "gates": {
        "single_qubit_time": 50e-9,  # s
        "two_qubit_time": 200e-9,  # s
        "measurement_time": 1e-6,  # s
        "gate_fidelity": 0.999,  # Typical gate fidelity
    },
}


# Conversion utilities
def temperature_to_energy(T: float, unit: str = "eV") -> float:
    """Convert temperature to energy."""
    E_J = PHYSICAL_CONSTANTS["kb"] * T
    if unit == "eV":
        return E_J / PHYSICAL_CONSTANTS["eV_to_J"]
    elif unit == "meV":
        return E_J / PHYSICAL_CONSTANTS["meV_to_J"]
    else:
        return E_J


def energy_to_temperature(E: float, unit: str = "eV") -> float:
    """Convert energy to temperature."""
    if unit == "eV":
        E_J = E * PHYSICAL_CONSTANTS["eV_to_J"]
    elif unit == "meV":
        E_J = E * PHYSICAL_CONSTANTS["meV_to_J"]
    else:
        E_J = E
    return E_J / PHYSICAL_CONSTANTS["kb"]


def planckian_time(T: float) -> float:
    """
    Calculate the Planckian dissipation time τ_P = ℏ/(kb⋅T).
    
    This is the fundamental quantum limit for scattering time in strange metals.
    """
    return PHYSICAL_CONSTANTS["hbar"] / (PHYSICAL_CONSTANTS["kb"] * T)


def strange_metal_resistivity(T: float, material_type: str = "cuprate") -> float:
    """
    Calculate T-linear resistivity characteristic of strange metals.
    
    ρ = ρ₀ + A⋅T where A ~ ℏ/(e²⋅a₀)
    
    Returns resistivity in μΩ⋅cm
    """
    # Universal strange metal coefficient
    A = MATERIAL_CONSTANTS["planckian"]["resistivity_coefficient"]  # μΩ⋅cm/K
    
    # Residual resistivity (material dependent)
    if material_type == "cuprate":
        rho_0 = 10.0  # μΩ⋅cm
    else:
        rho_0 = 20.0  # μΩ⋅cm
    
    return rho_0 + A * T


def coherence_length_bcs(gap: float, fermi_velocity: float) -> float:
    """
    BCS coherence length ξ = ℏ⋅v_F/(π⋅Δ).
    
    Args:
        gap: Superconducting gap in eV
        fermi_velocity: Fermi velocity in m/s
    
    Returns:
        Coherence length in meters
    """
    gap_J = gap * PHYSICAL_CONSTANTS["eV_to_J"]
    return PHYSICAL_CONSTANTS["hbar"] * fermi_velocity / (np.pi * gap_J)


def penetration_depth_london(carrier_density: float, 
                            effective_mass: float = 1.0) -> float:
    """
    London penetration depth λ = √(m/(μ₀⋅n⋅e²)).
    
    Args:
        carrier_density: Carrier density in m^-3
        effective_mass: Effective mass in units of electron mass
    
    Returns:
        Penetration depth in meters
    """
    m = effective_mass * PHYSICAL_CONSTANTS["me"]
    e = PHYSICAL_CONSTANTS["e"]
    mu_0 = PHYSICAL_CONSTANTS["mu_0"]
    
    return np.sqrt(m / (mu_0 * carrier_density * e**2))


def rydberg_blockade_radius(C6: Optional[float] = None,
                           Omega: Optional[float] = None) -> float:
    """
    Calculate Rydberg blockade radius r_b = (C6/Ω)^(1/6).
    
    Args:
        C6: Van der Waals coefficient (default from constants)
        Omega: Rabi frequency (default from constants)
    
    Returns:
        Blockade radius in meters
    """
    if C6 is None:
        C6 = SIMULATION_CONSTANTS["rydberg"]["C6"]
    if Omega is None:
        Omega = SIMULATION_CONSTANTS["rydberg"]["Omega_max"]
    
    return (C6 / Omega)**(1/6)


def pasqal_to_real_scale(pasqal_distance: float, 
                        real_lattice_constant: float = 3.8e-10) -> float:
    """
    Convert Pasqal simulation scale to real material scale.
    
    Args:
        pasqal_distance: Distance in Pasqal units (meters, typically μm)
        real_lattice_constant: Real material lattice constant (meters)
    
    Returns:
        Scaled distance in real material units
    """
    pasqal_unit = SIMULATION_CONSTANTS["pasqal"]["min_spacing"]
    scale_factor = real_lattice_constant / pasqal_unit
    return pasqal_distance * scale_factor


# Useful combinations
DERIVED_CONSTANTS = {
    # Quantum of resistance
    "R_Q": PHYSICAL_CONSTANTS["hbar"] / PHYSICAL_CONSTANTS["e"]**2,  # ~25.8 kΩ
    
    # Quantum of conductance  
    "G_Q": PHYSICAL_CONSTANTS["e"]**2 / PHYSICAL_CONSTANTS["hbar"],  # ~38.7 μS
    
    # Josephson constant
    "K_J": 2 * PHYSICAL_CONSTANTS["e"] / PHYSICAL_CONSTANTS["hbar"],  # ~483.6 THz/V
    
    # Compton wavelength
    "lambda_C": PHYSICAL_CONSTANTS["hbar"] / (PHYSICAL_CONSTANTS["me"] * PHYSICAL_CONSTANTS["c"]),  # ~2.43 pm
    
    # Rydberg constant
    "R_inf": PHYSICAL_CONSTANTS["me"] * PHYSICAL_CONSTANTS["e"]**4 / (
        8 * PHYSICAL_CONSTANTS["epsilon_0"]**2 * PHYSICAL_CONSTANTS["hbar"]**3 * PHYSICAL_CONSTANTS["c"]
    ),  # ~1.097e7 m^-1
}


# Temperature scales
TEMPERATURE_SCALES = {
    "room_temperature": 300,  # K
    "liquid_nitrogen": 77,  # K  
    "liquid_helium": 4.2,  # K
    "tc_ybco": 92,  # K
    "tc_bscco": 90,  # K
    "tc_hgbacuo": 135,  # K (under pressure)
    "tc_h3s": 203,  # K (high pressure superconductor)
}

# Individual constants for easy import
hbar = PHYSICAL_CONSTANTS["hbar"]
kb = PHYSICAL_CONSTANTS["kb"] 
e = PHYSICAL_CONSTANTS["e"]
me = PHYSICAL_CONSTANTS["me"]