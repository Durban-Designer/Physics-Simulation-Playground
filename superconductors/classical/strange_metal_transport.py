"""
Classical transport calculations for strange metals and superconductors.

This module implements the classical component of the tri-hybrid approach,
computing transport properties from analog and quantum simulation results
to predict superconducting behavior and T-linear resistivity.

Based on Boltzmann transport theory with quantum corrections from entanglement.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Callable
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize, curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Import core modules
try:
    from ..core.materials import Material, get_material
    from ..core.disorder import DisorderPattern
    from ..core.constants import PHYSICAL_CONSTANTS, kb, hbar, e
except ImportError:
    from core.materials import Material, get_material
    from core.disorder import DisorderPattern
    from core.constants import PHYSICAL_CONSTANTS, kb, hbar, e

print("✅ Classical transport module loaded")


@dataclass
class TransportParameters:
    """Parameters for transport calculations."""
    
    # Drude model parameters
    drude_scattering_rate: float = 1e12  # Hz, disorder scattering rate
    plasma_frequency: float = 1e15  # Hz, plasma frequency
    
    # Strange metal parameters
    planckian_coefficient: float = 1.0  # Coefficient in ρ = ρ₀ + α(kBT/ℏ)
    inelastic_scattering_rate: float = 1e11  # Hz, T-linear scattering
    
    # Superconducting parameters
    pairing_strength: float = 1.0  # Dimensionless pairing interaction
    coherence_length: float = 10.0  # nm, superconducting coherence length
    
    # Quantum correction parameters  
    entanglement_coupling: float = 0.1  # How entanglement affects transport


class ResistivityModel:
    """
    Models for calculating resistivity from microscopic parameters.
    
    Implements various models including Drude, strange metal T-linear,
    and quantum-corrected transport with entanglement effects.
    """
    
    def __init__(self, material: Union[str, Material],
                 transport_params: Optional[TransportParameters] = None):
        """
        Initialize resistivity model.
        
        Args:
            material: Material for calculations
            transport_params: Transport parameters
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.params = transport_params or TransportParameters()
    
    def drude_resistivity(self, temperature: float, 
                         disorder_strength: float) -> float:
        """
        Calculate Drude resistivity with disorder scattering.
        
        ρ = m/(ne²τ) where τ is disorder-dependent scattering time
        
        Args:
            temperature: Temperature in Kelvin
            disorder_strength: Disorder strength parameter
            
        Returns:
            Resistivity in Ω⋅m
        """
        # Disorder-dependent scattering rate
        gamma_disorder = self.params.drude_scattering_rate * (1 + disorder_strength)
        
        # Temperature-dependent phonon scattering
        gamma_phonon = 1e10 * (temperature / 300.0)**2  # T² phonon scattering
        
        # Total scattering rate
        gamma_total = gamma_disorder + gamma_phonon
        
        # Drude resistivity
        carrier_density = self.material.carrier_density  # electrons/m³
        resistivity = (PHYSICAL_CONSTANTS['me'] * gamma_total) / \
                     (carrier_density * e**2)
        
        return resistivity
    
    def strange_metal_resistivity(self, temperature: float,
                                 disorder_strength: float,
                                 entanglement_entropy: float = 0.0) -> float:
        """
        Calculate T-linear strange metal resistivity.
        
        ρ = ρ₀ + α(kBT/ℏ) with quantum entanglement corrections
        
        Args:
            temperature: Temperature in Kelvin
            disorder_strength: Disorder strength
            entanglement_entropy: Quantum entanglement entropy
            
        Returns:
            Resistivity in Ω⋅m
        """
        # Residual resistivity from disorder
        rho_0 = self.drude_resistivity(0.1, disorder_strength)  # Low-T limit
        
        # T-linear coefficient from Planckian scattering
        alpha = self.material.planckian_coefficient
        
        # Quantum correction from entanglement
        quantum_correction = self.params.entanglement_coupling * entanglement_entropy
        alpha_eff = alpha * (1 + quantum_correction)
        
        # T-linear resistivity
        t_linear_term = alpha_eff * (kb * temperature / hbar)
        
        resistivity = rho_0 + t_linear_term
        
        return resistivity
    
    def superconducting_resistivity(self, temperature: float,
                                  disorder_strength: float,
                                  order_parameter: float) -> float:
        """
        Calculate resistivity in superconducting state.
        
        Uses Bardeen-Cooper-Schrieffer theory with disorder effects.
        
        Args:
            temperature: Temperature in Kelvin
            disorder_strength: Disorder strength
            order_parameter: Superconducting order parameter
            
        Returns:
            Resistivity in Ω⋅m (approaches 0 for T < Tc)
        """
        # Superconducting gap
        gap = order_parameter * kb * self.material.tc_pristine
        
        if temperature >= self.material.tc_pristine:
            # Above Tc - normal state
            return self.strange_metal_resistivity(temperature, disorder_strength)
        
        # Below Tc - activated behavior
        if gap > 0:
            thermal_activation = np.exp(-gap / (kb * temperature))
            
            # Disorder reduces gap and increases residual resistivity
            disorder_suppression = 1 - 0.1 * disorder_strength
            gap_eff = gap * max(0, disorder_suppression)
            
            # Superconducting resistivity
            rho_normal = self.strange_metal_resistivity(self.material.tc_pristine, disorder_strength)
            rho_sc = rho_normal * thermal_activation * (1 + disorder_strength)
            
            return rho_sc
        else:
            # No superconductivity
            return self.strange_metal_resistivity(temperature, disorder_strength)
    
    def calculate_tc_suppression(self, disorder_strength: float,
                               entanglement_entropy: float = 0.0) -> float:
        """
        Calculate how disorder and entanglement affect Tc.
        
        Uses Abrikosov-Gor'kov theory with quantum corrections.
        
        Args:
            disorder_strength: Disorder strength
            entanglement_entropy: Quantum entanglement entropy
            
        Returns:
            Suppressed Tc in Kelvin
        """
        tc_pristine = self.material.tc_pristine
        
        # Disorder suppression (Abrikosov-Gor'kov)
        # Tc(disorder) = Tc₀ * sqrt(1 - γ²/Δ₀²)
        disorder_suppression = max(0, 1 - 0.5 * disorder_strength**2)
        
        # Quantum enhancement from optimal entanglement
        # Hypothesis: entanglement can enhance pairing if optimal
        optimal_entanglement = 1.0  # Optimal value
        entanglement_factor = 1 + 0.1 * np.exp(-(entanglement_entropy - optimal_entanglement)**2)
        
        tc_suppressed = tc_pristine * disorder_suppression * entanglement_factor
        
        return max(0, tc_suppressed)


class TransportCalculator:
    """
    Comprehensive transport property calculator.
    
    Combines results from analog and quantum simulations to compute
    resistivity, conductivity, and superconducting properties.
    """
    
    def __init__(self, material: Union[str, Material],
                 transport_params: Optional[TransportParameters] = None):
        """
        Initialize transport calculator.
        
        Args:
            material: Material for calculations
            transport_params: Transport parameters
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.params = transport_params or TransportParameters()
        self.resistivity_model = ResistivityModel(self.material, self.params)
    
    def compute_transport_properties(self, 
                                   analog_data: Dict,
                                   quantum_data: Optional[Dict] = None) -> Dict:
        """
        Compute comprehensive transport properties from simulation data.
        
        Args:
            analog_data: Results from analog simulation
            quantum_data: Results from quantum simulation (optional)
            
        Returns:
            Dictionary with transport properties
        """
        temperature = analog_data['temperature']
        disorder_strength = analog_data['disorder_metrics']['total']
        order_parameter = analog_data['mean_order']
        
        # Get quantum data if available
        entanglement_entropy = 0.0
        if quantum_data is not None:
            entanglement_entropy = quantum_data.get('entanglement_analysis', {}).get(
                'total_entanglement_entropy', 0.0
            )
        
        # Calculate resistivities
        drude_rho = self.resistivity_model.drude_resistivity(temperature, disorder_strength)
        strange_metal_rho = self.resistivity_model.strange_metal_resistivity(
            temperature, disorder_strength, entanglement_entropy
        )
        sc_rho = self.resistivity_model.superconducting_resistivity(
            temperature, disorder_strength, order_parameter
        )
        
        # Determine which model applies
        if self.material.is_in_strange_metal_regime(temperature):
            primary_resistivity = strange_metal_rho
            transport_regime = "strange_metal"
        elif temperature < self.material.tc_pristine and order_parameter > 0.1:
            primary_resistivity = sc_rho
            transport_regime = "superconducting"
        else:
            primary_resistivity = drude_rho
            transport_regime = "normal_metal"
        
        # Calculate derived properties
        conductivity = 1.0 / primary_resistivity if primary_resistivity > 0 else np.inf
        
        # Tc suppression
        tc_suppressed = self.resistivity_model.calculate_tc_suppression(
            disorder_strength, entanglement_entropy
        )
        
        return {
            'temperature': temperature,
            'resistivity': primary_resistivity,
            'conductivity': conductivity,
            'drude_resistivity': drude_rho,
            'strange_metal_resistivity': strange_metal_rho,
            'superconducting_resistivity': sc_rho,
            'transport_regime': transport_regime,
            'disorder_strength': disorder_strength,
            'entanglement_entropy': entanglement_entropy,
            'tc_suppressed': tc_suppressed,
            'tc_enhancement': tc_suppressed / self.material.tc_pristine,
            'order_parameter': order_parameter
        }
    
    def fit_t_linear_resistivity(self, temperatures: np.ndarray,
                               resistivities: np.ndarray) -> Dict[str, float]:
        """
        Fit T-linear resistivity model to data.
        
        ρ(T) = ρ₀ + αT
        
        Args:
            temperatures: Temperature array in Kelvin
            resistivities: Resistivity array in Ω⋅m
            
        Returns:
            Fit parameters and quality metrics
        """
        def linear_model(T, rho_0, alpha):
            return rho_0 + alpha * T
        
        # Fit linear model
        popt, pcov = curve_fit(linear_model, temperatures, resistivities)
        rho_0, alpha = popt
        
        # Calculate fit quality
        fit_values = linear_model(temperatures, rho_0, alpha)
        r_squared = 1 - np.sum((resistivities - fit_values)**2) / \
                       np.sum((resistivities - np.mean(resistivities))**2)
        
        # Convert α to Planckian units
        alpha_planckian = alpha * hbar / kb  # ℏα/kB
        
        return {
            'rho_0': rho_0,
            'alpha': alpha,
            'alpha_planckian': alpha_planckian,
            'r_squared': r_squared,
            'fit_temperatures': temperatures,
            'fit_resistivities': fit_values
        }


class PhaseDiagramMapper:
    """
    Maps phase diagrams in temperature-disorder space.
    
    Identifies regions of strange metal behavior, superconductivity,
    and normal metallic behavior based on simulation results.
    """
    
    def __init__(self, material: Union[str, Material]):
        """
        Initialize phase diagram mapper.
        
        Args:
            material: Material for phase diagram
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.transport_calc = TransportCalculator(self.material)
    
    def map_phase_boundaries(self, 
                           temperature_range: Tuple[float, float],
                           disorder_range: Tuple[float, float],
                           resolution: Tuple[int, int] = (20, 20)) -> Dict:
        """
        Map phase boundaries in T-disorder space.
        
        Args:
            temperature_range: (T_min, T_max) in Kelvin
            disorder_range: (disorder_min, disorder_max)
            resolution: (n_temps, n_disorders) grid resolution
            
        Returns:
            Phase diagram data
        """
        n_temps, n_disorders = resolution
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_temps)
        disorders = np.linspace(disorder_range[0], disorder_range[1], n_disorders)
        
        # Initialize phase map
        phase_map = np.zeros((n_temps, n_disorders), dtype=int)
        resistivity_map = np.zeros((n_temps, n_disorders))
        tc_map = np.zeros((n_temps, n_disorders))
        
        # Phase codes: 0=normal, 1=strange_metal, 2=superconducting
        for i, T in enumerate(temperatures):
            for j, disorder in enumerate(disorders):
                
                # Mock analog data for phase mapping
                analog_data = {
                    'temperature': T,
                    'disorder_metrics': {'total': disorder},
                    'mean_order': max(0, 1 - T/self.material.tc_pristine - disorder)
                }
                
                # Calculate transport properties
                transport = self.transport_calc.compute_transport_properties(analog_data)
                
                # Classify phase
                if transport['transport_regime'] == 'superconducting':
                    phase_map[i, j] = 2
                elif transport['transport_regime'] == 'strange_metal':
                    phase_map[i, j] = 1
                else:
                    phase_map[i, j] = 0
                
                resistivity_map[i, j] = transport['resistivity']
                tc_map[i, j] = transport['tc_suppressed']
        
        return {
            'temperatures': temperatures,
            'disorders': disorders,
            'phase_map': phase_map,
            'resistivity_map': resistivity_map,
            'tc_map': tc_map,
            'phase_labels': {0: 'Normal Metal', 1: 'Strange Metal', 2: 'Superconductor'}
        }
    
    def find_optimal_disorder(self, target_temperature: float,
                            disorder_range: Tuple[float, float] = (0.0, 1.0),
                            optimization_target: str = 'tc_enhancement') -> Dict:
        """
        Find optimal disorder level for maximizing superconducting properties.
        
        Args:
            target_temperature: Target operating temperature in Kelvin
            disorder_range: Range of disorder to search
            optimization_target: 'tc_enhancement' or 'conductivity'
            
        Returns:
            Optimization results
        """
        def objective_function(disorder):
            """Objective function for optimization."""
            analog_data = {
                'temperature': target_temperature,
                'disorder_metrics': {'total': disorder[0]},
                'mean_order': max(0, 1 - target_temperature/self.material.tc_pristine - disorder[0])
            }
            
            transport = self.transport_calc.compute_transport_properties(analog_data)
            
            if optimization_target == 'tc_enhancement':
                # Maximize Tc enhancement
                return -transport['tc_enhancement']  # Negative for minimization
            elif optimization_target == 'conductivity':
                # Maximize conductivity at target temperature
                return -transport['conductivity']
            else:
                raise ValueError(f"Unknown optimization target: {optimization_target}")
        
        # Optimize
        result = minimize(
            objective_function,
            x0=[(disorder_range[0] + disorder_range[1]) / 2],
            bounds=[disorder_range],
            method='L-BFGS-B'
        )
        
        optimal_disorder = result.x[0]
        
        # Calculate properties at optimal disorder
        optimal_analog_data = {
            'temperature': target_temperature,
            'disorder_metrics': {'total': optimal_disorder},
            'mean_order': max(0, 1 - target_temperature/self.material.tc_pristine - optimal_disorder)
        }
        
        optimal_transport = self.transport_calc.compute_transport_properties(optimal_analog_data)
        
        return {
            'optimal_disorder': optimal_disorder,
            'optimization_target': optimization_target,
            'target_temperature': target_temperature,
            'optimal_value': -result.fun,
            'transport_properties': optimal_transport,
            'optimization_success': result.success
        }


class StrangeMetalTransport:
    """
    Complete transport simulation workflow for strange metals.
    
    Integrates analog, quantum, and classical calculations to compute
    transport properties and identify optimal conditions for superconductivity.
    """
    
    def __init__(self, material: Union[str, Material]):
        """
        Initialize strange metal transport calculator.
        
        Args:
            material: Material for transport calculations
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.transport_calc = TransportCalculator(self.material)
        self.phase_mapper = PhaseDiagramMapper(self.material)
        
        # Results storage
        self.transport_results = []
    
    def analyze_tri_hybrid_results(self, 
                                 analog_results: List[Dict],
                                 quantum_results: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze complete tri-hybrid simulation results.
        
        Args:
            analog_results: Results from analog simulations
            quantum_results: Results from quantum simulations (optional)
            
        Returns:
            Comprehensive transport analysis
        """
        transport_data = []
        
        for i, analog_result in enumerate(analog_results):
            quantum_result = quantum_results[i] if quantum_results else None
            
            transport = self.transport_calc.compute_transport_properties(
                analog_result, quantum_result
            )
            transport_data.append(transport)
        
        self.transport_results = transport_data
        
        # Extract temperature-dependent data
        temperatures = np.array([t['temperature'] for t in transport_data])
        resistivities = np.array([t['resistivity'] for t in transport_data])
        conductivities = np.array([t['conductivity'] for t in transport_data])
        tc_values = np.array([t['tc_suppressed'] for t in transport_data])
        
        # Fit T-linear resistivity in strange metal regime
        strange_metal_mask = np.array([
            self.material.is_in_strange_metal_regime(T) for T in temperatures
        ])
        
        t_linear_fit = None
        if np.any(strange_metal_mask) and np.sum(strange_metal_mask) >= 2:
            t_linear_fit = self.transport_calc.fit_t_linear_resistivity(
                temperatures[strange_metal_mask],
                resistivities[strange_metal_mask]
            )
        
        # Find maximum Tc
        max_tc_idx = np.argmax(tc_values)
        optimal_conditions = transport_data[max_tc_idx]
        
        return {
            'temperatures': temperatures,
            'resistivities': resistivities,
            'conductivities': conductivities,
            'tc_values': tc_values,
            't_linear_fit': t_linear_fit,
            'optimal_conditions': optimal_conditions,
            'max_tc_enhancement': np.max(tc_values) / self.material.tc_pristine,
            'transport_data': transport_data,
            'material': self.material.name
        }
    
    def compute_full_phase_diagram(self, 
                                 temperature_range: Tuple[float, float] = None,
                                 disorder_range: Tuple[float, float] = (0.0, 0.5)) -> Dict:
        """
        Compute complete phase diagram for the material.
        
        Args:
            temperature_range: Temperature range in Kelvin
            disorder_range: Disorder strength range
            
        Returns:
            Complete phase diagram
        """
        if temperature_range is None:
            tc = self.material.tc_pristine
            temperature_range = (0.1 * tc, 2.0 * tc)
        
        phase_diagram = self.phase_mapper.map_phase_boundaries(
            temperature_range, disorder_range
        )
        
        return phase_diagram
    
    def discover_optimal_superconductor(self, 
                                      target_temp: float = 300.0,
                                      max_disorder: float = 0.3) -> Dict:
        """
        Discover optimal disorder engineering for room-temperature superconductivity.
        
        Args:
            target_temp: Target operating temperature in Kelvin
            max_disorder: Maximum allowed disorder strength
            
        Returns:
            Discovery results with optimal conditions
        """
        # Find optimal disorder for Tc enhancement
        tc_optimization = self.phase_mapper.find_optimal_disorder(
            target_temp, (0.0, max_disorder), 'tc_enhancement'
        )
        
        # Find optimal disorder for conductivity
        conductivity_optimization = self.phase_mapper.find_optimal_disorder(
            target_temp, (0.0, max_disorder), 'conductivity'
        )
        
        # Combine results
        discovery_result = {
            'target_temperature': target_temp,
            'material': self.material.name,
            'tc_optimization': tc_optimization,
            'conductivity_optimization': conductivity_optimization,
            'achievable_tc': tc_optimization['transport_properties']['tc_suppressed'],
            'room_temp_superconductor': tc_optimization['transport_properties']['tc_suppressed'] > 300.0,
            'discovery_confidence': min(1.0, tc_optimization['transport_properties']['tc_enhancement'])
        }
        
        return discovery_result