"""
Digital quantum circuits for strange metal entanglement simulation.

This module implements the quantum entanglement component of the tri-hybrid approach,
creating long-range entangled states that, when combined with atomic disorder,
produce the T-linear resistivity characteristic of strange metals.

Based on the Patel et al. (2023) mechanism: quantum entanglement + disorder = strange metal behavior
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import warnings

# Import quantum circuit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entanglement_of_formation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler

# Import core modules
try:
    from ..core.materials import Material, get_material
    from ..core.disorder import DisorderPattern, DisorderType
    from ..core.constants import SIMULATION_CONSTANTS
except ImportError:
    from core.materials import Material, get_material
    from core.disorder import DisorderPattern, DisorderType
    from core.constants import SIMULATION_CONSTANTS

print("✅ Quantum entanglement module loaded")


@dataclass
class EntanglementParameters:
    """Parameters controlling quantum entanglement generation."""
    
    entanglement_depth: int = 3  # Circuit depth for entanglement generation
    coupling_strength: float = 1.0  # Strength of entangling interactions
    decoherence_time: float = 100.0  # Decoherence time in μs
    interaction_range: float = 5.0  # Range of interactions in lattice units
    temperature_scaling: float = 0.01  # How temperature affects entanglement


class QuantumEntanglementCircuit:
    """
    Builds quantum circuits for strange metal entanglement simulation.
    
    Creates parameterized circuits that generate long-range entangled states
    characteristic of strange metals, with disorder-dependent parameters.
    """
    
    def __init__(self, material: Union[str, Material], 
                 disorder_pattern: Optional[DisorderPattern] = None,
                 entanglement_params: Optional[EntanglementParameters] = None):
        """
        Initialize quantum circuit builder.
        
        Args:
            material: Material for simulation
            disorder_pattern: Disorder configuration
            entanglement_params: Entanglement generation parameters
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.disorder = disorder_pattern or DisorderPattern()
        self.ent_params = entanglement_params or EntanglementParameters()
        
        # Track circuits for reuse
        self._circuit_cache = {}
    
    def create_entangling_circuit(self, n_qubits: int, 
                                 positions: np.ndarray,
                                 temperature: float = 100.0) -> QuantumCircuit:
        """
        Create parameterized circuit for generating entangled strange metal state.
        
        Args:
            n_qubits: Number of qubits (corresponds to lattice sites)
            positions: Physical positions of qubits/atoms
            temperature: Temperature in Kelvin
            
        Returns:
            Parameterized quantum circuit
        """
        # Create quantum and classical registers
        qreg = QuantumRegister(n_qubits, 'q')
        creg = ClassicalRegister(n_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Count interaction pairs within range first
        interaction_pairs = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= self.ent_params.interaction_range:
                    interaction_pairs.append((i, j))
        
        n_interactions = len(interaction_pairs) * self.ent_params.entanglement_depth
        
        # Create parameter vectors for disorder-dependent interactions
        theta_params = ParameterVector('θ', length=n_qubits)
        phi_params = ParameterVector('φ', length=max(1, n_interactions))  # Ensure at least 1
        
        # Initial state preparation - disorder-dependent single-qubit rotations
        for i in range(n_qubits):
            # Local disorder affects initial state preparation
            circuit.ry(theta_params[i], qreg[i])
        
        # Generate entanglement layers
        param_idx = 0
        for layer in range(self.ent_params.entanglement_depth):
            
            # Long-range entangling gates based on physical positions
            for i, j in interaction_pairs:
                if param_idx < len(phi_params.params):
                    # Parameterized entangling gate
                    circuit.rzz(phi_params[param_idx], qreg[i], qreg[j])
                    param_idx += 1
            
            # Single-qubit rotations between layers
            for i in range(n_qubits):
                circuit.rx(theta_params[i] * 0.5, qreg[i])
        
        # Final measurements
        circuit.measure_all()
        
        return circuit
    
    def calculate_circuit_parameters(self, positions: np.ndarray,
                                   temperature: float,
                                   disorder_strength: float) -> Dict[str, np.ndarray]:
        """
        Calculate parameter values for the quantum circuit based on disorder and temperature.
        
        Args:
            positions: Qubit positions
            temperature: Temperature in Kelvin
            disorder_strength: Disorder strength metric
            
        Returns:
            Dictionary of parameter values
        """
        n_qubits = len(positions)
        
        # Temperature-dependent scaling
        temp_factor = temperature / 300.0  # Normalized to room temperature
        
        # Single-qubit parameters - disorder creates local variations
        theta_values = []
        for i, pos in enumerate(positions):
            # Local disorder affects rotation angles
            local_disorder = self._calculate_local_disorder(pos, positions, disorder_strength)
            theta = np.pi/4 * (1.0 + 0.2 * local_disorder * temp_factor)
            theta_values.append(theta)
        
        # Two-qubit parameters - long-range interactions
        phi_values = []
        interaction_pairs = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                distance = np.linalg.norm(positions[i] - positions[j])
                
                if distance <= self.ent_params.interaction_range:
                    interaction_pairs.append((i, j))
                    # Distance and disorder dependent coupling
                    coupling = self.ent_params.coupling_strength / (1 + distance)
                    disorder_mod = 1.0 + 0.1 * disorder_strength * np.sin(distance)
                    phi = coupling * disorder_mod * temp_factor
                    phi_values.append(phi)
        
        # Extend phi_values for all layers
        n_layers = self.ent_params.entanglement_depth
        extended_phi_values = []
        for layer in range(n_layers):
            for phi in phi_values:
                extended_phi_values.append(phi * (1 + 0.1 * layer))  # Layer-dependent variation
        
        # Ensure we have at least one value
        if not extended_phi_values:
            extended_phi_values = [0.1]
        
        phi_values = extended_phi_values
        
        return {
            'θ': np.array(theta_values),
            'φ': np.array(phi_values)
        }
    
    def _calculate_local_disorder(self, position: np.ndarray, 
                                 all_positions: np.ndarray,
                                 global_disorder: float) -> float:
        """Calculate local disorder strength at a given position."""
        # Use correlation with nearby sites
        local_disorder = global_disorder
        
        for other_pos in all_positions:
            if not np.array_equal(position, other_pos):
                distance = np.linalg.norm(position - other_pos)
                correlation_length = self.disorder.correlation_length
                
                # Exponential correlation
                correlation = np.exp(-distance / correlation_length)
                local_disorder += 0.1 * global_disorder * correlation * np.random.normal(0, 0.1)
        
        return np.clip(local_disorder, 0, 2.0)  # Keep reasonable bounds


class EntanglementAnalyzer:
    """
    Analyzes entanglement properties of quantum states from strange metal simulation.
    
    Computes entanglement entropy, pairwise entanglement, and other measures
    relevant to understanding the quantum nature of strange metals.
    """
    
    def __init__(self):
        """Initialize entanglement analyzer."""
        pass
    
    def compute_entanglement_entropy(self, statevector: Statevector,
                                   subsystem_qubits: List[int]) -> float:
        """
        Compute von Neumann entanglement entropy for a subsystem.
        
        Args:
            statevector: Full system quantum state
            subsystem_qubits: List of qubit indices for subsystem
            
        Returns:
            Entanglement entropy in bits
        """
        # Get reduced density matrix for subsystem
        rho_sub = partial_trace(statevector, subsystem_qubits)
        
        # Compute eigenvalues
        if hasattr(rho_sub.data, 'toarray'):
            eigenvalues = np.real(np.linalg.eigvals(rho_sub.data.toarray()))
        else:
            eigenvalues = np.real(np.linalg.eigvals(rho_sub.data))
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Remove numerical zeros
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return entropy
    
    def compute_pairwise_entanglement(self, statevector: Statevector,
                                    n_qubits: int) -> np.ndarray:
        """
        Compute pairwise entanglement between all pairs of qubits.
        
        Args:
            statevector: Full system quantum state
            n_qubits: Total number of qubits
            
        Returns:
            Matrix of pairwise entanglement measures
        """
        entanglement_matrix = np.zeros((n_qubits, n_qubits))
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Trace out all qubits except i and j
                other_qubits = [k for k in range(n_qubits) if k not in [i, j]]
                
                if other_qubits:
                    rho_ij = partial_trace(statevector, other_qubits)
                    
                    # Use entanglement of formation
                    entanglement = entanglement_of_formation(rho_ij)
                    entanglement_matrix[i, j] = entanglement
                    entanglement_matrix[j, i] = entanglement
        
        return entanglement_matrix
    
    def analyze_entanglement_structure(self, statevector: Statevector,
                                     positions: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Comprehensive analysis of entanglement structure in the quantum state.
        
        Args:
            statevector: Quantum state to analyze
            positions: Physical positions of qubits
            
        Returns:
            Dictionary with entanglement analysis results
        """
        n_qubits = len(positions)
        
        # Global entanglement measures
        total_entropy = self.compute_entanglement_entropy(statevector, list(range(n_qubits // 2)))
        
        # Pairwise entanglement
        pairwise_ent = self.compute_pairwise_entanglement(statevector, n_qubits)
        
        # Spatial correlation of entanglement
        distances = []
        entanglements = []
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                distance = np.linalg.norm(positions[i] - positions[j])
                distances.append(distance)
                entanglements.append(pairwise_ent[i, j])
        
        distances = np.array(distances)
        entanglements = np.array(entanglements)
        
        # Sort by distance for correlation analysis
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_entanglements = entanglements[sorted_indices]
        
        return {
            'total_entanglement_entropy': total_entropy,
            'pairwise_entanglement_matrix': pairwise_ent,
            'mean_pairwise_entanglement': np.mean(entanglements),
            'max_pairwise_entanglement': np.max(entanglements),
            'distance_entanglement_correlation': np.corrcoef(distances, entanglements)[0, 1],
            'distances': sorted_distances,
            'entanglements': sorted_entanglements
        }


class StrangeMetalQuantumSimulation:
    """
    Complete quantum simulation workflow for strange metals.
    
    Coordinates circuit construction, parameter calculation, execution,
    and entanglement analysis for strange metal physics.
    """
    
    def __init__(self, material: Union[str, Material],
                 disorder_pattern: Optional[DisorderPattern] = None,
                 entanglement_params: Optional[EntanglementParameters] = None):
        """
        Initialize quantum simulation.
        
        Args:
            material: Material for simulation
            disorder_pattern: Disorder configuration
            entanglement_params: Entanglement parameters
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.disorder = disorder_pattern or DisorderPattern()
        self.ent_params = entanglement_params or EntanglementParameters()
        
        # Initialize components
        self.circuit_builder = QuantumEntanglementCircuit(
            self.material, self.disorder, self.ent_params
        )
        self.entanglement_analyzer = EntanglementAnalyzer()
        
        # Quantum backend
        self.backend = AerSimulator()
        
        # Results storage
        self.simulation_results = []
    
    def run_quantum_simulation(self, positions: np.ndarray,
                             temperature: float,
                             n_shots: int = 1000) -> Dict:
        """
        Run complete quantum entanglement simulation.
        
        Args:
            positions: Physical positions of quantum sites
            temperature: Temperature in Kelvin
            n_shots: Number of measurement shots
            
        Returns:
            Dictionary with simulation results
        """
        n_qubits = len(positions)
        
        # Calculate disorder strength
        disorder_metrics = self.disorder.compute_disorder_strength(positions)
        disorder_strength = disorder_metrics['total']
        
        # Create quantum circuit
        circuit = self.circuit_builder.create_entangling_circuit(
            n_qubits, positions, temperature
        )
        
        # Calculate parameter values
        param_values = self.circuit_builder.calculate_circuit_parameters(
            positions, temperature, disorder_strength
        )
        
        # Bind parameters to circuit
        bound_circuit = circuit.assign_parameters({
            **{f'θ[{i}]': param_values['θ'][i] for i in range(len(param_values['θ']))},
            **{f'φ[{i}]': param_values['φ'][i] for i in range(len(param_values['φ']))}
        })
        
        # Transpile circuit
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        transpiled_circuit = pass_manager.run(bound_circuit)
        
        # Run simulation for measurements
        sampler = StatevectorSampler()
        job = sampler.run([transpiled_circuit], shots=n_shots)
        measurement_results = job.result()[0]
        
        # Get statevector for entanglement analysis
        statevector_circuit = transpiled_circuit.remove_final_measurements(inplace=False)
        statevector = Statevector.from_instruction(statevector_circuit)
        
        # Analyze entanglement
        entanglement_analysis = self.entanglement_analyzer.analyze_entanglement_structure(
            statevector, positions
        )
        
        # Package results
        result = {
            'temperature': temperature,
            'n_qubits': n_qubits,
            'positions': positions,
            'disorder_strength': disorder_strength,
            'circuit_depth': transpiled_circuit.depth(),
            'measurement_counts': measurement_results.data['c'].get_counts(),
            'n_shots': n_shots,
            'entanglement_analysis': entanglement_analysis,
            'quantum_state': statevector,
            'simulation_method': 'quantum_entanglement'
        }
        
        return result
    
    def scan_temperature_entanglement(self, positions: np.ndarray,
                                    temp_range: Tuple[float, float],
                                    n_temps: int = 10,
                                    **kwargs) -> List[Dict]:
        """
        Scan temperature to study entanglement evolution in strange metals.
        
        Args:
            positions: Physical positions of quantum sites
            temp_range: (T_min, T_max) in Kelvin
            n_temps: Number of temperature points
            **kwargs: Additional arguments for run_quantum_simulation
            
        Returns:
            List of results at each temperature
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], n_temps)
        results = []
        
        for T in temperatures:
            print(f"Running quantum simulation at T = {T:.1f}K...")
            result = self.run_quantum_simulation(positions, T, **kwargs)
            results.append(result)
            
            # Check if in strange metal regime
            if self.material.is_in_strange_metal_regime(T):
                result['in_strange_metal_regime'] = True
            else:
                result['in_strange_metal_regime'] = False
        
        self.simulation_results = results
        return results
    
    def extract_quantum_transport_data(self) -> Dict[str, np.ndarray]:
        """
        Extract quantum transport-relevant data for classical calculations.
        
        Returns:
            Dictionary with quantum data for transport calculations
        """
        if not self.simulation_results:
            raise ValueError("No simulation results available. Run quantum simulation first.")
        
        temperatures = np.array([r['temperature'] for r in self.simulation_results])
        entanglement_entropies = np.array([
            r['entanglement_analysis']['total_entanglement_entropy'] 
            for r in self.simulation_results
        ])
        mean_entanglements = np.array([
            r['entanglement_analysis']['mean_pairwise_entanglement']
            for r in self.simulation_results
        ])
        disorder_strengths = np.array([r['disorder_strength'] for r in self.simulation_results])
        
        # Calculate quantum coherence measures
        coherence_lengths = []
        for result in self.simulation_results:
            # Use entanglement decay with distance as proxy for coherence length
            distances = result['entanglement_analysis']['distances']
            entanglements = result['entanglement_analysis']['entanglements']
            
            # Fit exponential decay to get coherence length
            if len(distances) > 1:
                # Simple estimate: distance where entanglement drops to 1/e
                max_ent = np.max(entanglements)
                target_ent = max_ent / np.e
                
                # Find closest point
                idx = np.argmin(np.abs(entanglements - target_ent))
                coherence_length = distances[idx] if idx < len(distances) else distances[-1]
            else:
                coherence_length = 1.0
            
            coherence_lengths.append(coherence_length)
        
        return {
            'temperatures': temperatures,
            'entanglement_entropies': entanglement_entropies,
            'mean_entanglements': mean_entanglements,
            'disorder_strengths': disorder_strengths,
            'quantum_coherence_lengths': np.array(coherence_lengths),
            'quantum_correlation_function': mean_entanglements * np.exp(-disorder_strengths)
        }
    
    def compute_entanglement_scaling(self, max_qubits: int = 8,
                                   temperature: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Study how entanglement scales with system size.
        
        Args:
            max_qubits: Maximum number of qubits to test
            temperature: Temperature for scaling study
            
        Returns:
            Dictionary with scaling results
        """
        qubit_counts = range(2, max_qubits + 1)
        entanglement_entropies = []
        circuit_depths = []
        
        for n_qubits in qubit_counts:
            # Create simple linear chain positions
            positions = np.array([[i * 5.0, 0.0] for i in range(n_qubits)])
            
            result = self.run_quantum_simulation(positions, temperature, n_shots=100)
            
            entanglement_entropies.append(
                result['entanglement_analysis']['total_entanglement_entropy']
            )
            circuit_depths.append(result['circuit_depth'])
        
        return {
            'n_qubits': np.array(qubit_counts),
            'entanglement_entropies': np.array(entanglement_entropies),
            'circuit_depths': np.array(circuit_depths)
        }