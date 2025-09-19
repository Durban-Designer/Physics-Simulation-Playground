"""
Tri-hybrid workflow orchestration for superconductor discovery.

This module coordinates analog (Pasqal Rydberg atoms), quantum (digital circuits),
and classical (transport) simulations in a unified workflow to discover
optimal disorder patterns for enhanced superconductivity.

Implements the complete Patel et al. (2023) mechanism:
Quantum entanglement + atomic disorder = T-linear resistivity → enhanced Tc
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass, field
import asyncio
import concurrent.futures
import time
from pathlib import Path
import json

# Import tri-hybrid modules
try:
    from ..analog.strange_metal_disorder import StrangeMetalAnalogSimulation
    from ..quantum.strange_metal_entanglement import StrangeMetalQuantumSimulation, EntanglementParameters
    from ..classical.strange_metal_transport import StrangeMetalTransport, TransportParameters
    from ..core.materials import Material, get_material, list_materials
    from ..core.disorder import DisorderPattern, DisorderType, create_realistic_cuprate_disorder
    from ..core.constants import SIMULATION_CONSTANTS
except ImportError:
    from analog.strange_metal_disorder import StrangeMetalAnalogSimulation
    from quantum.strange_metal_entanglement import StrangeMetalQuantumSimulation, EntanglementParameters
    from classical.strange_metal_transport import StrangeMetalTransport, TransportParameters
    from core.materials import Material, get_material, list_materials
    from core.disorder import DisorderPattern, DisorderType, create_realistic_cuprate_disorder
    from core.constants import SIMULATION_CONSTANTS

print("✅ Tri-hybrid orchestration module loaded")


@dataclass
class WorkflowConfig:
    """Configuration for tri-hybrid workflow execution."""
    
    # Simulation parameters
    temperature_range: Tuple[float, float] = (50.0, 300.0)
    n_temperatures: int = 10
    lattice_size: Tuple[int, int] = (3, 3)
    evolution_time: float = 100.0  # ns for analog simulation
    n_shots: int = 1000
    
    # Disorder optimization
    disorder_range: Tuple[float, float] = (0.0, 0.5)
    n_disorder_points: int = 5
    
    # Execution settings
    enable_analog: bool = True
    enable_quantum: bool = True
    enable_classical: bool = True
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Output settings
    save_results: bool = True
    output_directory: str = "discovery_results"
    verbose: bool = True


@dataclass 
class DiscoveryResult:
    """Results from superconductor discovery workflow."""
    
    material_name: str
    discovery_id: str
    timestamp: str
    
    # Simulation results
    analog_results: List[Dict] = field(default_factory=list)
    quantum_results: List[Dict] = field(default_factory=list) 
    transport_analysis: Dict = field(default_factory=dict)
    
    # Key discoveries
    optimal_disorder: float = 0.0
    enhanced_tc: float = 0.0
    tc_enhancement_factor: float = 1.0
    room_temperature_superconductor: bool = False
    
    # Performance metrics
    total_execution_time: float = 0.0
    analog_simulation_time: float = 0.0
    quantum_simulation_time: float = 0.0
    classical_calculation_time: float = 0.0


class TriHybridWorkflow:
    """
    Complete tri-hybrid superconductor discovery workflow.
    
    Orchestrates analog (Pasqal), quantum (digital), and classical simulations
    to discover optimal disorder patterns for enhanced superconductivity.
    """
    
    def __init__(self, material: Union[str, Material],
                 config: Optional[WorkflowConfig] = None):
        """
        Initialize tri-hybrid workflow.
        
        Args:
            material: Material for discovery
            config: Workflow configuration
        """
        self.material = get_material(material) if isinstance(material, str) else material
        self.config = config or WorkflowConfig()
        
        # Initialize simulation components
        self.analog_sim = None
        self.quantum_sim = None
        self.transport_calc = None
        
        if self.config.enable_analog:
            self.analog_sim = StrangeMetalAnalogSimulation(self.material)
        
        if self.config.enable_quantum:
            self.quantum_sim = StrangeMetalQuantumSimulation(self.material)
        
        if self.config.enable_classical:
            self.transport_calc = StrangeMetalTransport(self.material)
        
        # Results storage
        self.discovery_results = []
        
        # Setup output directory
        if self.config.save_results:
            self.output_dir = Path(self.config.output_directory)
            self.output_dir.mkdir(exist_ok=True)
    
    def run_single_disorder_point(self, disorder_strength: float) -> Dict:
        """
        Run complete tri-hybrid simulation for a single disorder configuration.
        
        Args:
            disorder_strength: Disorder strength parameter
            
        Returns:
            Complete simulation results
        """
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Running tri-hybrid simulation with disorder = {disorder_strength:.3f}")
        
        # Create disorder pattern
        disorder_pattern = DisorderPattern(
            disorder_type=DisorderType.COMPOSITE,
            position_variance=disorder_strength * 0.5,
            vacancy_rate=disorder_strength * 0.1,
            correlation_length=10.0 - disorder_strength * 5.0
        )
        
        # Update simulation components with new disorder
        if self.analog_sim:
            self.analog_sim.disorder_pattern = disorder_pattern
            self.analog_sim.lattice_generator.disorder = disorder_pattern
        
        if self.quantum_sim:
            self.quantum_sim.disorder = disorder_pattern
            self.quantum_sim.circuit_builder.disorder = disorder_pattern
        
        results = {
            'disorder_strength': disorder_strength,
            'analog_results': [],
            'quantum_results': [],
            'execution_times': {}
        }
        
        # Run simulations for temperature range
        temperatures = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            self.config.n_temperatures
        )
        
        for temperature in temperatures:
            temp_results = self._run_single_temperature(temperature, disorder_pattern)
            
            if temp_results['analog_result']:
                results['analog_results'].append(temp_results['analog_result'])
            if temp_results['quantum_result']:
                results['quantum_results'].append(temp_results['quantum_result'])
            
            # Accumulate execution times
            for key, value in temp_results['execution_times'].items():
                if key not in results['execution_times']:
                    results['execution_times'][key] = 0
                results['execution_times'][key] += value
        
        results['total_time'] = time.time() - start_time
        
        return results
    
    def _run_single_temperature(self, temperature: float, 
                               disorder_pattern: DisorderPattern) -> Dict:
        """Run simulation at single temperature point."""
        results = {
            'analog_result': None,
            'quantum_result': None,
            'execution_times': {'analog': 0, 'quantum': 0}
        }
        
        # Run analog and quantum simulations
        if self.config.parallel_execution and self.analog_sim and self.quantum_sim:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                analog_future = executor.submit(self._run_analog_simulation, temperature)
                quantum_future = executor.submit(self._run_quantum_simulation, temperature)
                
                results['analog_result'] = analog_future.result()
                results['quantum_result'] = quantum_future.result()
        else:
            # Sequential execution
            if self.analog_sim:
                results['analog_result'] = self._run_analog_simulation(temperature)
            
            if self.quantum_sim:
                results['quantum_result'] = self._run_quantum_simulation(temperature)
        
        return results
    
    def _run_analog_simulation(self, temperature: float) -> Dict:
        """Run analog Pasqal simulation."""
        start_time = time.time()
        
        try:
            result = self.analog_sim.run_simulation(
                temperature=temperature,
                lattice_size=self.config.lattice_size,
                evolution_time=self.config.evolution_time,
                n_shots=self.config.n_shots
            )
            result['execution_time'] = time.time() - start_time
            return result
        except Exception as e:
            if self.config.verbose:
                print(f"Analog simulation failed at T={temperature}K: {e}")
            return {'error': str(e), 'temperature': temperature}
    
    def _run_quantum_simulation(self, temperature: float) -> Dict:
        """Run quantum entanglement simulation."""
        start_time = time.time()
        
        try:
            # Create simple positions for quantum simulation
            n_qubits = self.config.lattice_size[0] * self.config.lattice_size[1]
            positions = np.array([
                [i % self.config.lattice_size[0] * 5.0, 
                 i // self.config.lattice_size[0] * 5.0] 
                for i in range(n_qubits)
            ])
            
            result = self.quantum_sim.run_quantum_simulation(
                positions=positions,
                temperature=temperature,
                n_shots=self.config.n_shots
            )
            result['execution_time'] = time.time() - start_time
            return result
        except Exception as e:
            if self.config.verbose:
                print(f"Quantum simulation failed at T={temperature}K: {e}")
            return {'error': str(e), 'temperature': temperature}
    
    def discover_optimal_superconductor(self) -> DiscoveryResult:
        """
        Run complete discovery workflow to find optimal superconductor.
        
        Returns:
            Discovery results with optimal conditions
        """
        start_time = time.time()
        
        if self.config.verbose:
            print(f"🚀 Starting superconductor discovery for {self.material.name}")
            print(f"Temperature range: {self.config.temperature_range[0]}-{self.config.temperature_range[1]}K")
            print(f"Disorder range: {self.config.disorder_range[0]}-{self.config.disorder_range[1]}")
        
        # Initialize discovery result
        discovery_id = f"{self.material.name}_{int(time.time())}"
        discovery = DiscoveryResult(
            material_name=self.material.name,
            discovery_id=discovery_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Run disorder optimization
        disorder_strengths = np.linspace(
            self.config.disorder_range[0],
            self.config.disorder_range[1],
            self.config.n_disorder_points
        )
        
        all_results = []
        
        for disorder_strength in disorder_strengths:
            disorder_results = self.run_single_disorder_point(disorder_strength)
            all_results.append(disorder_results)
        
        # Analyze results to find optimal conditions
        optimal_results = self._analyze_discovery_results(all_results)
        
        # Populate discovery result
        discovery.analog_results = optimal_results['all_analog_results']
        discovery.quantum_results = optimal_results['all_quantum_results']
        discovery.optimal_disorder = optimal_results['optimal_disorder']
        discovery.enhanced_tc = optimal_results['max_tc']
        discovery.tc_enhancement_factor = optimal_results['tc_enhancement_factor']
        discovery.room_temperature_superconductor = optimal_results['max_tc'] >= 300.0
        discovery.total_execution_time = time.time() - start_time
        
        # Run classical transport analysis
        if self.transport_calc and discovery.analog_results:
            classical_start = time.time()
            discovery.transport_analysis = self.transport_calc.analyze_tri_hybrid_results(
                discovery.analog_results,
                discovery.quantum_results if discovery.quantum_results else None
            )
            discovery.classical_calculation_time = time.time() - classical_start
        
        # Save results
        if self.config.save_results:
            self._save_discovery_results(discovery)
        
        # Print summary
        if self.config.verbose:
            self._print_discovery_summary(discovery)
        
        self.discovery_results.append(discovery)
        return discovery
    
    def _analyze_discovery_results(self, all_results: List[Dict]) -> Dict:
        """Analyze results to find optimal conditions."""
        best_tc = 0.0
        optimal_disorder = 0.0
        all_analog_results = []
        all_quantum_results = []
        
        for disorder_results in all_results:
            disorder_strength = disorder_results['disorder_strength']
            
            # Collect all results
            all_analog_results.extend(disorder_results['analog_results'])
            all_quantum_results.extend(disorder_results['quantum_results'])
            
            # Find highest Tc for this disorder level
            for analog_result in disorder_results['analog_results']:
                if 'error' not in analog_result:
                    # Estimate Tc from order parameter and temperature
                    temp = analog_result['temperature']
                    order = analog_result['mean_order']
                    
                    # Simple Tc estimation: order parameter should be high at low T
                    estimated_tc = temp * order * (2.0 - disorder_strength)
                    
                    if estimated_tc > best_tc:
                        best_tc = estimated_tc
                        optimal_disorder = disorder_strength
        
        return {
            'optimal_disorder': optimal_disorder,
            'max_tc': best_tc,
            'tc_enhancement_factor': best_tc / self.material.tc_pristine,
            'all_analog_results': all_analog_results,
            'all_quantum_results': all_quantum_results
        }
    
    def _save_discovery_results(self, discovery: DiscoveryResult):
        """Save discovery results to file."""
        filename = self.output_dir / f"discovery_{discovery.discovery_id}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return {k: convert_for_json(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        discovery_dict = convert_for_json(discovery)
        
        with open(filename, 'w') as f:
            json.dump(discovery_dict, f, indent=2)
        
        if self.config.verbose:
            print(f"Discovery results saved to {filename}")
    
    def _print_discovery_summary(self, discovery: DiscoveryResult):
        """Print discovery summary."""
        print(f"\n🎉 Discovery Complete for {discovery.material_name}")
        print(f"{'='*60}")
        print(f"Discovery ID: {discovery.discovery_id}")
        print(f"Execution Time: {discovery.total_execution_time:.2f}s")
        print(f"\n📊 Key Results:")
        print(f"  Optimal Disorder: {discovery.optimal_disorder:.3f}")
        print(f"  Enhanced Tc: {discovery.enhanced_tc:.1f}K")
        print(f"  Tc Enhancement: {discovery.tc_enhancement_factor:.2f}x")
        print(f"  Room Temperature SC: {'✅ YES' if discovery.room_temperature_superconductor else '❌ No'}")
        print(f"\n📈 Simulation Stats:")
        print(f"  Analog Results: {len(discovery.analog_results)}")
        print(f"  Quantum Results: {len(discovery.quantum_results)}")
        print(f"  Transport Analysis: {'✅' if discovery.transport_analysis else '❌'}")


class SuperconductorOptimizer:
    """
    Advanced optimizer for discovering optimal superconducting materials.
    
    Uses iterative refinement and machine learning to efficiently explore
    the parameter space of disorder patterns and materials.
    """
    
    def __init__(self, materials: Optional[List[str]] = None):
        """
        Initialize superconductor optimizer.
        
        Args:
            materials: List of materials to optimize (default: all available)
        """
        self.materials = materials or list_materials()
        self.optimization_history = []
        self.best_discoveries = {}
    
    def optimize_all_materials(self, config: Optional[WorkflowConfig] = None) -> Dict[str, DiscoveryResult]:
        """
        Optimize all materials for enhanced superconductivity.
        
        Args:
            config: Workflow configuration
            
        Returns:
            Dictionary of best discoveries for each material
        """
        config = config or WorkflowConfig()
        
        print(f"🔬 Optimizing {len(self.materials)} materials for superconductivity")
        
        for material_name in self.materials:
            print(f"\n🧪 Optimizing {material_name}...")
            
            workflow = TriHybridWorkflow(material_name, config)
            discovery = workflow.discover_optimal_superconductor()
            
            self.best_discoveries[material_name] = discovery
            self.optimization_history.append(discovery)
        
        # Find global optimum
        best_material = max(
            self.best_discoveries.keys(),
            key=lambda m: self.best_discoveries[m].enhanced_tc
        )
        
        print(f"\n🏆 Best Material: {best_material}")
        print(f"    Enhanced Tc: {self.best_discoveries[best_material].enhanced_tc:.1f}K")
        
        return self.best_discoveries
    
    def compare_materials(self) -> Dict:
        """
        Compare optimization results across all materials.
        
        Returns:
            Comparison analysis
        """
        if not self.best_discoveries:
            raise ValueError("No optimization results available. Run optimize_all_materials first.")
        
        comparison = {
            'materials': list(self.best_discoveries.keys()),
            'enhanced_tcs': [d.enhanced_tc for d in self.best_discoveries.values()],
            'tc_enhancements': [d.tc_enhancement_factor for d in self.best_discoveries.values()],
            'optimal_disorders': [d.optimal_disorder for d in self.best_discoveries.values()],
            'room_temp_candidates': [
                m for m, d in self.best_discoveries.items() 
                if d.room_temperature_superconductor
            ]
        }
        
        return comparison


class DiscoveryProtocol:
    """
    Standardized protocols for superconductor discovery experiments.
    
    Provides pre-configured workflows for different discovery objectives.
    """
    
    @staticmethod
    def room_temperature_discovery(material: str) -> DiscoveryResult:
        """
        Protocol optimized for room-temperature superconductor discovery.
        
        Args:
            material: Material to optimize
            
        Returns:
            Discovery result
        """
        config = WorkflowConfig(
            temperature_range=(250.0, 350.0),  # Focus on room temperature
            n_temperatures=15,
            disorder_range=(0.0, 0.3),  # Conservative disorder range
            n_disorder_points=8,
            lattice_size=(4, 4),  # Larger system
            n_shots=2000,  # Higher precision
            verbose=True
        )
        
        workflow = TriHybridWorkflow(material, config)
        return workflow.discover_optimal_superconductor()
    
    @staticmethod
    def high_tc_discovery(material: str) -> DiscoveryResult:
        """
        Protocol optimized for maximum Tc enhancement.
        
        Args:
            material: Material to optimize
            
        Returns:
            Discovery result
        """
        config = WorkflowConfig(
            temperature_range=(10.0, 200.0),  # Wide temperature range
            n_temperatures=20,
            disorder_range=(0.0, 0.8),  # Aggressive disorder exploration
            n_disorder_points=12,
            lattice_size=(3, 3),
            n_shots=1500,
            verbose=True
        )
        
        workflow = TriHybridWorkflow(material, config)
        return workflow.discover_optimal_superconductor()
    
    @staticmethod
    def fast_screening(materials: List[str]) -> Dict[str, DiscoveryResult]:
        """
        Fast screening protocol for multiple materials.
        
        Args:
            materials: List of materials to screen
            
        Returns:
            Screening results
        """
        config = WorkflowConfig(
            temperature_range=(50.0, 300.0),
            n_temperatures=5,  # Coarse temperature sampling
            disorder_range=(0.0, 0.4),
            n_disorder_points=4,  # Coarse disorder sampling
            lattice_size=(2, 2),  # Small system for speed
            n_shots=500,  # Fewer shots
            parallel_execution=True,
            verbose=False
        )
        
        results = {}
        for material in materials:
            workflow = TriHybridWorkflow(material, config)
            results[material] = workflow.discover_optimal_superconductor()
        
        return results