"""
Material definitions for strange metal superconductors.

This module contains material parameter definitions for various superconducting
materials, particularly focusing on cuprates and iron-based superconductors that
exhibit strange metal behavior.

All material parameters are sourced from peer-reviewed literature.
See materials_citations.py for detailed references.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
from enum import Enum


class MaterialType(Enum):
    """Types of superconducting materials."""
    CUPRATE = "cuprate"
    IRON_BASED = "iron_based"
    NICKELATE = "nickelate"
    ORGANIC = "organic"
    HEAVY_FERMION = "heavy_fermion"


@dataclass
class Material:
    """Base class for superconducting materials with strange metal properties."""
    
    name: str
    material_type: MaterialType
    lattice_constant: float  # Angstroms
    
    # Electronic properties
    carrier_density: float  # electrons/cm^3
    effective_mass: float  # electron mass units
    fermi_energy: float  # eV
    
    # Superconducting properties
    tc_pristine: float  # K, pristine Tc
    gap_symmetry: str  # "s", "d", "p", etc.
    coherence_length: float  # Angstroms
    
    # Strange metal properties
    planckian_coefficient: float  # α in ρ = ρ₀ + αkB T/ℏ
    strange_metal_range: Tuple[float, float]  # Temperature range [T_min, T_max]
    resistivity_exponent: float  # n in ρ ∝ T^n (should be ~1 for strange metals)
    
    # Disorder properties
    optimal_disorder: Optional[float] = None  # Optimal disorder strength for max Tc
    disorder_sensitivity: float = 1.0  # How sensitive Tc is to disorder
    
    # Additional parameters
    additional_params: Dict[str, float] = field(default_factory=dict)
    
    def get_hopping_parameters(self) -> Dict[str, float]:
        """Calculate hopping parameters for tight-binding models."""
        t = self.fermi_energy / 4.0  # Nearest neighbor hopping
        return {
            't': t,
            't_prime': -0.3 * t,  # Next-nearest neighbor
            't_double_prime': 0.2 * t,  # Next-next-nearest
        }
    
    def get_interaction_strength(self) -> float:
        """Calculate electron-electron interaction strength U."""
        # Estimate U from resistivity coefficient
        return 4.0 * self.fermi_energy * self.planckian_coefficient
    
    def is_in_strange_metal_regime(self, temperature: float) -> bool:
        """Check if temperature is in strange metal regime."""
        return self.strange_metal_range[0] <= temperature <= self.strange_metal_range[1]


@dataclass
class CuprateMaterial(Material):
    """Cuprate superconductor specific properties."""
    
    # CuO2 plane structure
    cu_o_distance: float = 1.9  # Angstroms
    planar_spacing: float = 11.7  # Angstroms between CuO2 planes
    
    # Doping
    optimal_doping: float = 0.16  # Optimal hole doping
    doping_range: Tuple[float, float] = (0.05, 0.30)  # Superconducting dome
    
    # Competing phases
    antiferromagnetic_tn: float = 300.0  # Néel temperature at zero doping
    charge_order_onset: float = 0.12  # Doping where CDW appears
    pseudogap_temperature: Optional[float] = None
    
    def __post_init__(self):
        """Set cuprate-specific defaults."""
        self.material_type = MaterialType.CUPRATE
        if self.gap_symmetry is None:
            self.gap_symmetry = "d"  # d-wave pairing
        
        # Calculate pseudogap temperature if not provided
        if self.pseudogap_temperature is None:
            self.pseudogap_temperature = 1.5 * self.tc_pristine
    
    def get_oxygen_positions(self, nx: int, ny: int) -> List[Tuple[float, float]]:
        """Generate oxygen positions in CuO2 plane."""
        positions = []
        a = self.lattice_constant
        
        for i in range(nx):
            for j in range(ny):
                # Cu position
                cu_x, cu_y = i * a, j * a
                
                # O positions (between Cu atoms)
                positions.append((cu_x + a/2, cu_y))  # O_x
                positions.append((cu_x, cu_y + a/2))  # O_y
        
        return positions


@dataclass
class IronBasedMaterial(Material):
    """Iron-based superconductor specific properties."""
    
    # FeAs/FeSe layer structure
    fe_as_distance: float = 2.4  # Angstroms
    layer_spacing: float = 13.0  # Angstroms
    
    # Multi-orbital nature
    num_orbitals: int = 5  # Fe d-orbitals
    orbital_energies: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, -0.1, -0.2])
    
    # Magnetic properties
    spin_density_wave_temp: float = 150.0  # SDW transition
    magnetic_moment: float = 1.0  # Bohr magnetons
    
    # Nematicity
    nematic_temperature: Optional[float] = None
    orthorhombic_distortion: float = 0.003  # (a-b)/(a+b)
    
    def __post_init__(self):
        """Set iron-based specific defaults."""
        self.material_type = MaterialType.IRON_BASED
        if self.gap_symmetry is None:
            self.gap_symmetry = "s+-"  # s+- pairing
        
        if self.nematic_temperature is None:
            self.nematic_temperature = self.spin_density_wave_temp + 10


# Pre-defined materials database
MATERIALS_DATABASE = {
    # Cuprates
    "YBCO": CuprateMaterial(
        name="YBa2Cu3O7",
        material_type=MaterialType.CUPRATE,
        lattice_constant=3.82,
        carrier_density=1e21,
        effective_mass=3.0,
        fermi_energy=0.5,
        tc_pristine=92.0,
        gap_symmetry="d",
        coherence_length=15.0,
        planckian_coefficient=1.0,
        strange_metal_range=(100, 300),
        resistivity_exponent=1.0,
        optimal_disorder=0.05,
        disorder_sensitivity=0.8,
        additional_params={
            "apical_oxygen_distance": 2.4,
            "chain_site_occupancy": 1.0,
        }
    ),
    
    "BSCCO": CuprateMaterial(
        name="Bi2Sr2CaCu2O8",
        material_type=MaterialType.CUPRATE,
        lattice_constant=3.83,
        carrier_density=8e20,
        effective_mass=4.0,
        fermi_energy=0.4,
        tc_pristine=90.0,
        gap_symmetry="d",
        coherence_length=20.0,
        planckian_coefficient=1.1,
        strange_metal_range=(95, 300),
        resistivity_exponent=1.0,
        optimal_disorder=0.08,
        disorder_sensitivity=0.9,
        additional_params={
            "bismuth_layer_buckling": 0.2,
            "supermodulation_period": 4.7,
        }
    ),
    
    "LSCO": CuprateMaterial(
        name="La2-xSrxCuO4",
        material_type=MaterialType.CUPRATE,
        lattice_constant=3.78,
        carrier_density=5e20,
        effective_mass=3.5,
        fermi_energy=0.45,
        tc_pristine=38.0,
        gap_symmetry="d",
        coherence_length=25.0,
        planckian_coefficient=0.9,
        strange_metal_range=(40, 300),
        resistivity_exponent=1.0,
        optimal_disorder=0.03,
        disorder_sensitivity=0.7,
        optimal_doping=0.15,
        additional_params={
            "octahedral_tilt": 5.0,  # degrees
            "stripe_order_onset": 0.12,
        }
    ),
    
    "HgBaCuO": CuprateMaterial(
        name="HgBa2CuO4",
        material_type=MaterialType.CUPRATE,
        lattice_constant=3.87,
        carrier_density=1.2e21,
        effective_mass=2.5,
        fermi_energy=0.55,
        tc_pristine=98.0,
        gap_symmetry="d",
        coherence_length=12.0,
        planckian_coefficient=0.95,
        strange_metal_range=(100, 400),
        resistivity_exponent=1.0,
        optimal_disorder=0.04,
        disorder_sensitivity=0.85,
        additional_params={
            "hg_vacancy_concentration": 0.05,
            "pressure_coefficient": 2.0,  # K/GPa
        }
    ),
    
    # Iron-based superconductors
    "FeSe": IronBasedMaterial(
        name="FeSe",
        material_type=MaterialType.IRON_BASED,
        lattice_constant=3.77,
        carrier_density=3e20,
        effective_mass=3.0,
        fermi_energy=0.05,
        tc_pristine=8.0,
        gap_symmetry="s+-",
        coherence_length=30.0,
        planckian_coefficient=0.8,
        strange_metal_range=(10, 100),
        resistivity_exponent=1.0,
        optimal_disorder=0.02,
        disorder_sensitivity=1.2,
        fe_as_distance=2.37,
        nematic_temperature=155.0,
        additional_params={
            "pressure_induced_tc_max": 37.0,
            "monolayer_tc": 65.0,
        }
    ),
    
    "BaFeCoAs": IronBasedMaterial(
        name="Ba(Fe1-xCox)2As2",
        material_type=MaterialType.IRON_BASED,
        lattice_constant=3.96,
        carrier_density=5e20,
        effective_mass=3.5,
        fermi_energy=0.1,
        tc_pristine=25.0,
        gap_symmetry="s+-",
        coherence_length=20.0,
        planckian_coefficient=0.9,
        strange_metal_range=(30, 200),
        resistivity_exponent=1.0,
        optimal_disorder=0.06,
        disorder_sensitivity=1.0,
        spin_density_wave_temp=140.0,
        additional_params={
            "optimal_co_doping": 0.06,
            "quantum_critical_doping": 0.055,
        }
    ),
    
    "LaFeAsO": IronBasedMaterial(
        name="LaFeAsO1-xFx",
        material_type=MaterialType.IRON_BASED,
        lattice_constant=4.03,
        carrier_density=4e20,
        effective_mass=4.0,
        fermi_energy=0.08,
        tc_pristine=26.0,
        gap_symmetry="s+-",
        coherence_length=18.0,
        planckian_coefficient=1.0,
        strange_metal_range=(30, 200),
        resistivity_exponent=1.0,
        optimal_disorder=0.05,
        disorder_sensitivity=0.9,
        spin_density_wave_temp=150.0,
        additional_params={
            "optimal_f_doping": 0.11,
            "rare_earth_substitution_effect": 0.2,
        }
    ),
}


def get_material(name: str) -> Material:
    """Get material from database by name."""
    if name not in MATERIALS_DATABASE:
        raise ValueError(f"Material {name} not found in database. "
                        f"Available materials: {list(MATERIALS_DATABASE.keys())}")
    return MATERIALS_DATABASE[name]


def list_materials(material_type: Optional[MaterialType] = None) -> List[str]:
    """List available materials, optionally filtered by type."""
    if material_type is None:
        return list(MATERIALS_DATABASE.keys())
    
    return [name for name, mat in MATERIALS_DATABASE.items() 
            if mat.material_type == material_type]


def create_custom_material(name: str, base_material: str, **kwargs) -> Material:
    """Create a custom material based on an existing one with modifications."""
    base = get_material(base_material)
    
    # Create a copy with modified parameters
    material_dict = base.__dict__.copy()
    material_dict['name'] = name
    material_dict.update(kwargs)
    
    # Determine the correct class
    if isinstance(base, CuprateMaterial):
        return CuprateMaterial(**material_dict)
    elif isinstance(base, IronBasedMaterial):
        return IronBasedMaterial(**material_dict)
    else:
        return Material(**material_dict)