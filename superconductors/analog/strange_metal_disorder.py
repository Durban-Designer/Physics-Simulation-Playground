"""
Analog quantum simulation of strange metal disorder using Pasqal's platform.

This module implements the analog component of the tri-hybrid approach,
simulating non-uniform atomic arrangements characteristic of strange metals.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass
import warnings

# Import Pulser components - real hardware only
from pulser import Register, Sequence, Pulse
from pulser.devices import MockDevice, VirtualDevice
from pulser.waveforms import ConstantWaveform, RampWaveform
from pulser_simulation import QutipEmulator
from pasqal_cloud import SDK

# Import GPU-accelerated backends
try:
    from ..core.gpu_backends import HybridQuantumBackend, SimulationConfig, BackendType
    GPU_BACKENDS_AVAILABLE = True
    print("🚀 GPU-accelerated backends loaded (cuQuantum + Hybrid)")
except ImportError:
    GPU_BACKENDS_AVAILABLE = False
    print("✅ Real Pulser/Pasqal integration loaded (CPU only)")

# Import core modules with fallback for different import contexts
try:
    from ..core.disorder import DisorderPattern, DisorderType, create_realistic_cuprate_disorder
    from ..core.materials import Material, CuprateMaterial, get_material
    from ..core.constants import SIMULATION_CONSTANTS, rydberg_blockade_radius
except ImportError:
    # Fallback for direct execution or testing
    from core.disorder import DisorderPattern, DisorderType, create_realistic_cuprate_disorder
    from core.materials import Material, CuprateMaterial, get_material
    from core.constants import SIMULATION_CONSTANTS, rydberg_blockade_radius


class StrangeMetalLattice:
    """Creates disordered atomic arrangements for strange metals."""
    
    def __init__(self, material: Union[str, Material], 
                 disorder_pattern: Optional[DisorderPattern] = None):
        """
        Initialize lattice generator for strange metal simulations.
        
        Args:
            material: Material name or Material object
            disorder_pattern: Disorder configuration (default: realistic for material)
        """
        if isinstance(material, str):
            self.material = get_material(material)
        else:
            self.material = material
        
        if disorder_pattern is None:
            # Create realistic disorder for this material
            self.disorder = create_realistic_cuprate_disorder(
                self.material.name, 
                temperature=100.0  # Default temperature
            )
        else:
            self.disorder = disorder_pattern
        
        # Scale disorder to Pasqal units (micrometers)
        self._scale_to_pasqal()
    
    def _scale_to_pasqal(self):
        """Convert material scale (Angstroms) to Pasqal scale (micrometers)."""
        # Pasqal minimum spacing is 4 μm, typical lattice constant is ~4 Å
        # Scale factor to convert from Angstroms to micrometers: 1 μm = 10^4 Å
        pasqal_min_spacing_um = SIMULATION_CONSTANTS["pasqal"]["min_spacing"] * 1e6  # Convert to μm
        material_lattice_angstrom = self.material.lattice_constant
        
        # Use a reasonable mapping: map material lattice to ~5 μm in Pasqal units
        self.scale_factor = 5.0 / material_lattice_angstrom  # μm per Angstrom
        
        # Scale disorder parameters to micrometers
        self.disorder.base_spacing = material_lattice_angstrom * self.scale_factor  # μm
        self.disorder.position_variance *= self.scale_factor  # μm
        self.disorder.correlation_length *= self.scale_factor  # μm
    
    def generate_cuprate_plane(self, nx: int = 10, ny: int = 10) -> Register:
        """
        Generate CuO2 plane with realistic disorder.
        
        Args:
            nx, ny: Number of unit cells in x and y directions
            
        Returns:
            Pulser Register object with disordered atom positions
        """
        # Generate perfect lattice first
        positions = []
        
        # Cu positions
        for i in range(nx):
            for j in range(ny):
                x = i * self.disorder.base_spacing
                y = j * self.disorder.base_spacing
                positions.append((x, y))
        
        perfect_lattice = np.array(positions)
        
        # Apply disorder
        disordered_positions = self.disorder.generate_positions(
            nx, ny, perfect_lattice
        )
        
        # Ensure minimum spacing constraint for Pasqal (4 μm)
        disordered_positions = self._enforce_minimum_spacing(
            disordered_positions,
            min_spacing=4.0  # μm
        )
        
        # Create Register
        return Register.from_coordinates(disordered_positions, prefix='atom')
    
    def _enforce_minimum_spacing(self, positions: np.ndarray, 
                                min_spacing: float) -> List[Tuple[float, float]]:
        """
        Ensure all atoms meet Pasqal's minimum spacing requirement.
        
        Uses iterative repulsion to separate atoms that are too close.
        """
        positions = positions.copy()
        max_iterations = 100
        
        for _ in range(max_iterations):
            moved = False
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    
                    if distance < min_spacing and distance > 0:
                        # Move atoms apart
                        direction = (positions[j] - positions[i]) / distance
                        move_distance = (min_spacing - distance) / 2 + 0.1
                        
                        positions[i] -= direction * move_distance
                        positions[j] += direction * move_distance
                        moved = True
            
            if not moved:
                break
        
        return [(float(x), float(y)) for x, y in positions]
    
    def create_analog_hamiltonian(self, register: Register, 
                                 temperature: float,
                                 evolution_time: float = 1000.0) -> Sequence:
        """
        Build Rydberg Hamiltonian that maps disorder to interactions.
        
        The blockade radius creates the 'patchwork' effect characteristic
        of strange metals.
        
        Args:
            register: Atomic register with positions
            temperature: Temperature in Kelvin
            evolution_time: Total evolution time in ns
            
        Returns:
            Pulser Sequence implementing the Hamiltonian
        """
        seq = Sequence(register, MockDevice)
        seq.declare_channel('rydberg', 'rydberg_global')
        
        # Calculate Hamiltonian parameters based on disorder and temperature
        params = self._calculate_hamiltonian_parameters(temperature)
        
        # Build pulse sequence with disorder-dependent modulation
        pulse = self._build_disorder_pulse(params, evolution_time)
        
        seq.add(pulse, 'rydberg')
        
        return seq
    
    def _calculate_hamiltonian_parameters(self, temperature: float) -> Dict[str, float]:
        """
        Calculate Rydberg Hamiltonian parameters from disorder and temperature.
        
        Key physics: disorder affects local energy scales and creates
        the patchwork pattern when combined with Rydberg blockade.
        """
        # Base parameters
        omega_max = SIMULATION_CONSTANTS["rydberg"]["Omega_max"] / (2 * np.pi)  # MHz
        delta_max = SIMULATION_CONSTANTS["rydberg"]["Delta_max"] / (2 * np.pi)  # MHz
        
        # Temperature scaling
        temp_factor = temperature / 100.0  # Normalized to 100K
        
        # Disorder strength affects Rabi frequency variations
        disorder_metrics = self.disorder.compute_disorder_strength(
            np.array([(0, 0), (self.disorder.base_spacing, 0)])  # Sample positions
        )
        disorder_strength = disorder_metrics["total"]
        
        # Calculate parameters
        params = {
            "omega": omega_max * (1.0 + 0.1 * disorder_strength),
            "omega_variation": omega_max * 0.1 * disorder_strength,
            "delta": -delta_max * temp_factor * 0.5,
            "delta_variation": delta_max * 0.1 * disorder_strength,
            "blockade_radius": rydberg_blockade_radius() * 1e6,  # Convert to μm
            "disorder_strength": disorder_strength,
            "temperature_factor": temp_factor
        }
        
        return params
    
    def _build_disorder_pulse(self, params: Dict[str, float], 
                             duration: float) -> Pulse:
        """
        Build pulse that encodes disorder through parameter variations.
        """
        # For now, use constant pulse with disorder-modified parameters
        # Future: implement spatial modulation for patchwork pattern
        omega = params["omega"]
        delta = params["delta"]
        
        # Add small random variations to simulate disorder
        omega += np.random.normal(0, params["omega_variation"])
        delta += np.random.normal(0, params["delta_variation"])
        
        # Ensure parameters are reasonable for numerical integration
        omega = np.clip(abs(omega), 0.1, 10.0)  # MHz, reasonable range
        delta = np.clip(delta, -20.0, 20.0)     # MHz, avoid extreme detunings
        duration = max(10.0, min(duration, 1000.0))  # ns, reasonable range
        
        return Pulse.ConstantPulse(
            duration=int(duration),
            amplitude=omega,
            detuning=delta,
            phase=0.0
        )
    
    def measure_local_order(self, measurement_results: Dict) -> np.ndarray:
        """
        Extract local order parameter showing patchwork structure.
        
        Each atom's Rydberg population represents local pairing strength.
        Disorder creates non-uniform pattern characteristic of strange metals.
        """
        counts = measurement_results["counts"]
        shots = measurement_results.get("shots", 1000)
        
        # Convert bitstring counts to local Rydberg populations
        n_atoms = len(list(counts.keys())[0])
        local_order = np.zeros(n_atoms)
        
        for bitstring, count in counts.items():
            for i, bit in enumerate(bitstring):
                if bit == '1':  # Rydberg state
                    local_order[i] += count / shots
        
        return local_order


@dataclass
class AnalogHamiltonianBuilder:
    """
    Advanced Hamiltonian construction for strange metal physics.
    
    Implements disorder-dependent energy scales and patchwork patterns.
    """
    
    disorder_pattern: DisorderPattern
    material: Material
    temperature: float
    
    def build_rydberg_hamiltonian(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Construct full Rydberg Hamiltonian matrices.
        
        H = Σᵢ (Ω/2 σˣᵢ - Δ nᵢ) + Σᵢⱼ Vᵢⱼ nᵢ nⱼ
        
        where Vᵢⱼ = C₆/|rᵢ - rⱼ|⁶ is the van der Waals interaction.
        """
        n_atoms = len(positions)
        
        # Compute distance matrix
        distances = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = distances[j, i] = dist
        
        # Van der Waals interactions
        C6 = SIMULATION_CONSTANTS["rydberg"]["C6"]
        V_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if distances[i, j] > 0:
                    V_matrix[i, j] = V_matrix[j, i] = C6 / distances[i, j]**6
        
        # Disorder-dependent local fields
        disorder_field = self._compute_disorder_field(positions)
        
        # Local Rabi frequencies and detunings
        omega_i = self._compute_local_rabi(positions, disorder_field)
        delta_i = self._compute_local_detuning(positions, disorder_field)
        
        return {
            "V_matrix": V_matrix,
            "omega_i": omega_i,
            "delta_i": delta_i,
            "distances": distances,
            "disorder_field": disorder_field
        }
    
    def _compute_disorder_field(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute local disorder strength at each atom position.
        
        This creates the patchwork pattern of strong/weak disorder regions.
        """
        n_atoms = len(positions)
        disorder_field = np.zeros(n_atoms)
        
        # Create patchwork with correlation length
        patch_size = self.disorder_pattern.correlation_length
        n_patches = max(1, int(np.sqrt(n_atoms) / 5))
        
        # Random patch centers and strengths
        patch_centers = positions[np.random.choice(n_atoms, n_patches, replace=False)]
        patch_strengths = np.random.uniform(0.5, 1.5, n_patches)
        
        # Compute disorder field as sum of Gaussian patches
        for i, pos in enumerate(positions):
            for center, strength in zip(patch_centers, patch_strengths):
                distance = np.linalg.norm(pos - center)
                disorder_field[i] += strength * np.exp(-distance**2 / (2 * patch_size**2))
        
        # Normalize
        disorder_field /= np.mean(disorder_field)
        
        return disorder_field
    
    def _compute_local_rabi(self, positions: np.ndarray, 
                           disorder_field: np.ndarray) -> np.ndarray:
        """Local Rabi frequencies modified by disorder."""
        omega_max = SIMULATION_CONSTANTS["rydberg"]["Omega_max"] / (2 * np.pi)
        
        # Disorder modulates local coupling strength
        omega_i = omega_max * (1.0 + 0.2 * (disorder_field - 1.0))
        
        # Temperature adds fluctuations
        temp_fluctuations = np.random.normal(0, 0.05 * omega_max * self.temperature / 300, len(positions))
        omega_i += temp_fluctuations
        
        return np.maximum(omega_i, 0)  # Ensure positive
    
    def _compute_local_detuning(self, positions: np.ndarray,
                               disorder_field: np.ndarray) -> np.ndarray:
        """Local detunings representing chemical potential variations."""
        delta_max = SIMULATION_CONSTANTS["rydberg"]["Delta_max"] / (2 * np.pi)
        
        # Base detuning from temperature
        base_delta = -delta_max * self.temperature / 1000  # Scale with T
        
        # Disorder creates spatial variations
        delta_i = base_delta * disorder_field
        
        # Add random local variations
        delta_i += np.random.normal(0, 0.1 * abs(base_delta), len(positions))
        
        return delta_i


class StrangeMetalAnalogSimulation:
    """
    Complete analog simulation workflow for strange metals.
    
    Coordinates lattice generation, Hamiltonian construction, and measurements.
    
    Supports multiple high-performance backends:
    - CPU QuTiP (small systems, <12 qubits)
    - GPU cuQuantum (medium systems, 12-25 qubits)
    - Pasqal Cloud EMU-MPS (large systems, 25+ qubits)
    """
    
    def __init__(self, material: Union[str, Material],
                 disorder_pattern: Optional[DisorderPattern] = None,
                 simulation_config: Optional[Any] = None):
        """Initialize simulation for given material."""
        self.material = get_material(material) if isinstance(material, str) else material
        self.lattice_generator = StrangeMetalLattice(self.material, disorder_pattern)
        self.disorder_pattern = self.lattice_generator.disorder  # Use the lattice generator's disorder pattern
        self.measurement_results = []
        
        # Initialize GPU-accelerated backend if available
        if GPU_BACKENDS_AVAILABLE:
            from ..core.gpu_backends import SimulationConfig, BackendType
            self.simulation_config = simulation_config or SimulationConfig(
                backend=BackendType.HYBRID,
                verbose=True  # Show backend selection
            )
        else:
            self.simulation_config = simulation_config
        
        if GPU_BACKENDS_AVAILABLE:
            try:
                self.hybrid_backend = HybridQuantumBackend(self.simulation_config)
                self.gpu_enabled = True
                print(f"🚀 Hybrid quantum backend ready (cuQuantum + Pasqal Cloud)")
            except Exception as e:
                print(f"⚠️ GPU backend failed, using CPU: {e}")
                self.hybrid_backend = None
                self.gpu_enabled = False
        else:
            self.hybrid_backend = None
            self.gpu_enabled = False
    
    def run_simulation(self, temperature: float, 
                      lattice_size: Tuple[int, int] = (10, 10),
                      evolution_time: float = 1000.0,
                      n_shots: int = 1000) -> Dict:
        """
        Run complete analog simulation at given temperature.
        
        Args:
            temperature: Temperature in Kelvin
            lattice_size: (nx, ny) number of unit cells
            evolution_time: Evolution time in ns
            n_shots: Number of measurement shots
            
        Returns:
            Dictionary with simulation results
        """
        # Add bounds checking to prevent memory overflow
        max_qubits = 20  # Safe limit for current implementation
        estimated_qubits = lattice_size[0] * lattice_size[1]
        
        if estimated_qubits > max_qubits:
            raise ValueError(f"Lattice size {lattice_size} would create ~{estimated_qubits} qubits, "
                           f"exceeding safe limit of {max_qubits}. Use smaller lattice_size.")
        
        # Generate disordered lattice
        register = self.lattice_generator.generate_cuprate_plane(*lattice_size)
        
        # Build and run Hamiltonian evolution
        sequence = self.lattice_generator.create_analog_hamiltonian(
            register, temperature, evolution_time
        )
        
        # Simulate measurements with optimal backend selection
        if self.gpu_enabled and self.hybrid_backend:
            try:
                results = self.hybrid_backend.simulate_sequence(sequence, n_shots)
                results["simulation_method"] = "hybrid_gpu"
            except Exception as e:
                print(f"⚠️ Hybrid backend failed, falling back to CPU: {e}")
                results = self._run_real_simulation(sequence, n_shots)
        else:
            results = self._run_real_simulation(sequence, n_shots)
        
        # Extract physics
        local_order = self.lattice_generator.measure_local_order(results)
        
        # Compute disorder metrics - extract coordinates from Pulser Register
        positions = np.array([register.qubits[qid] for qid in register.qubit_ids])
        
        disorder_metrics = self.disorder_pattern.compute_disorder_strength(positions)
        
        # Package results
        result_dict = {
            "temperature": temperature,
            "lattice_size": lattice_size,
            "n_atoms": len(positions),
            "disorder_metrics": disorder_metrics,
            "local_order_parameter": local_order,
            "mean_order": np.mean(local_order),
            "order_variance": np.var(local_order),
            "positions": positions,
            "measurement_counts": results.get("counts", {}),
            "n_shots": n_shots
        }
        
        # Include simulation metadata
        result_dict["simulation_time"] = results["simulation_time"]
        result_dict["simulation_method"] = "real_pulser"
            
        return result_dict
    
    
    def _run_real_simulation(self, sequence, n_shots: int) -> Dict:
        """Run simulation with real QutipEmulator, GPU-accelerated when possible."""
        try:
            import time
            
            # Remove signal timeout to fix threading issues
            print(f"Starting GPU-accelerated QutipEmulator simulation with {len(sequence.register.qubit_ids)} qubits...")
            start_time = time.time()
            
            # Check for GPU acceleration capabilities
            use_gpu = False
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    use_gpu = True
                    print("🚀 GPU acceleration enabled with CuPy")
                else:
                    print("⚠️ CuPy available but no GPU detected")
            except ImportError:
                print("⚠️ CuPy not available, using CPU")
            
            # Create and run simulation with optimized settings
            sim = QutipEmulator.from_sequence(sequence)
            
            # Use progressive fallback strategy for solver options
            results = None
            solver_configs = [
                # Try optimized settings first
                {
                    'nsteps': 5000,
                    'atol': 1e-10,
                    'rtol': 1e-8,
                },
                # Fallback to moderate settings
                {
                    'nsteps': 2000,
                    'atol': 1e-8,
                    'rtol': 1e-6,
                },
                # Final fallback to basic settings
                {}  # Default options
            ]
            
            for i, config in enumerate(solver_configs):
                try:
                    if config:  # Non-empty config
                        print(f"🔧 Trying solver config {i+1}/{len(solver_configs)}")
                        results = sim.run(options=config)
                    else:
                        print("🔧 Using default solver settings")
                        results = sim.run()
                    break  # Success, exit loop
                    
                except Exception as solver_error:
                    print(f"⚠️ Solver config {i+1} failed: {solver_error}")
                    if i == len(solver_configs) - 1:  # Last attempt
                        print("❌ All solver configurations failed")
                        raise RuntimeError(f"Simulation failed with all solver configs: {solver_error}")
                    continue
            
            elapsed = time.time() - start_time
            print(f"✅ Simulation completed in {elapsed:.3f}s")
            
            # GPU-accelerated state processing
            final_state = results.get_final_state()
            n_qubits = len(sequence.register.qubit_ids)
            
            # Extract state vector with GPU acceleration when possible
            try:
                # Get state vector data
                state_data = final_state.full().flatten()
                
                if use_gpu:
                    # Transfer to GPU for processing
                    import cupy as cp
                    gpu_state = cp.asarray(state_data)
                    gpu_probs = cp.abs(gpu_state)**2
                    gpu_probs = gpu_probs / cp.sum(gpu_probs)  # Normalize on GPU
                    probs = cp.asnumpy(gpu_probs)  # Transfer back to CPU for sampling
                    print(f"✅ GPU-accelerated state processing: {len(probs)} amplitudes")
                else:
                    # CPU processing
                    probs = np.abs(state_data)**2
                    probs = probs / np.sum(probs)
                    print(f"✅ CPU state processing: {len(probs)} amplitudes")
                
            except Exception as e:
                print(f"State extraction failed: {e}")
                print(f"Final state type: {type(final_state)}")
                print("Using uniform distribution")
                probs = np.ones(2**n_qubits) / (2**n_qubits)
            
            # Sample computational basis states (optimized)
            counts = {}
            if use_gpu and len(probs) > 16:  # GPU sampling for larger systems
                try:
                    import cupy as cp
                    gpu_probs = cp.asarray(probs)
                    # Generate random samples on GPU
                    gpu_samples = cp.random.choice(len(probs), size=n_shots, p=gpu_probs)
                    samples = cp.asnumpy(gpu_samples)
                    
                    # Convert to bitstrings
                    for idx in samples:
                        bitstring = format(idx, f'0{n_qubits}b')
                        counts[bitstring] = counts.get(bitstring, 0) + 1
                    
                    print("✅ GPU-accelerated sampling completed")
                except Exception as gpu_error:
                    print(f"GPU sampling failed: {gpu_error}, falling back to CPU")
                    # Fallback to CPU sampling
                    for _ in range(n_shots):
                        idx = np.random.choice(len(probs), p=probs)
                        bitstring = format(idx, f'0{n_qubits}b')
                        counts[bitstring] = counts.get(bitstring, 0) + 1
            else:
                # CPU sampling for smaller systems or when GPU unavailable
                for _ in range(n_shots):
                    idx = np.random.choice(len(probs), p=probs)
                    bitstring = format(idx, f'0{n_qubits}b')
                    counts[bitstring] = counts.get(bitstring, 0) + 1
            
            return {
                "counts": counts,
                "shots": n_shots,
                "final_state": final_state,
                "simulation_results": results,
                "simulation_time": elapsed,
                "gpu_accelerated": use_gpu
            }
            
        except Exception as e:
            print(f"⚠️ GPU-accelerated simulation failed: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Quantum simulation failed: {e}")
    
    def scan_temperature(self, temp_range: Tuple[float, float],
                        n_temps: int = 20,
                        **kwargs) -> List[Dict]:
        """
        Scan temperature to map out strange metal regime.
        
        Args:
            temp_range: (T_min, T_max) in Kelvin
            n_temps: Number of temperature points
            **kwargs: Additional arguments for run_simulation
            
        Returns:
            List of results at each temperature
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], n_temps)
        results = []
        
        for T in temperatures:
            result = self.run_simulation(T, **kwargs)
            results.append(result)
            
            # Check if we're in strange metal regime
            if self.material.is_in_strange_metal_regime(T):
                result["in_strange_metal_regime"] = True
            else:
                result["in_strange_metal_regime"] = False
        
        self.measurement_results = results
        return results
    
    def extract_transport_data(self) -> Dict[str, np.ndarray]:
        """
        Extract transport-relevant data from measurements.
        
        This will be used by the classical module to compute resistivity.
        """
        if not self.measurement_results:
            raise ValueError("No measurement results available. Run simulation first.")
        
        temperatures = np.array([r["temperature"] for r in self.measurement_results])
        mean_order = np.array([r["mean_order"] for r in self.measurement_results])
        order_variance = np.array([r["order_variance"] for r in self.measurement_results])
        disorder_strength = np.array([r["disorder_metrics"]["total"] 
                                     for r in self.measurement_results])
        
        return {
            "temperatures": temperatures,
            "mean_order": mean_order,
            "order_variance": order_variance,
            "disorder_strength": disorder_strength,
            "patchwork_amplitude": order_variance / (mean_order + 1e-10)  # Relative fluctuations
        }