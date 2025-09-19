#!/usr/bin/env python3
"""
Personal Superconductor Research Script

Uses existing modules to run experiments with local GPU and Pasqal Cloud integration.
Results are stored in PostgreSQL for analysis via the Next.js web interface.
"""

import os
import sys
import time
import json
import psycopg2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules for tri-hybrid integration
from core.materials import get_material, CuprateMaterial, IronBasedMaterial
from core.disorder import create_realistic_cuprate_disorder, DisorderPattern
from analog.strange_metal_disorder import StrangeMetalAnalogSimulation
from quantum.strange_metal_entanglement import StrangeMetalQuantumSimulation
from classical.strange_metal_transport import StrangeMetalTransport
from orchestration.strange_metal_workflow import TriHybridWorkflow, WorkflowConfig
from core.gpu_backends import SimulationConfig, BackendType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Simple PostgreSQL connection for storing experiment results."""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL', 
            'postgresql://postgres:research123@localhost:5432/experiments'
        )
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def insert_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Insert experiment result into database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experiments (
                        material, material_formula, temperature_k, disorder_strength,
                        n_qubits, backend, simulation_time_seconds, result,
                        cost_euros, shots, notes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    experiment_data['material'],
                    experiment_data.get('material_formula'),
                    experiment_data['temperature_k'],
                    experiment_data['disorder_strength'],
                    experiment_data['n_qubits'],
                    experiment_data['backend'],
                    experiment_data.get('simulation_time_seconds'),
                    json.dumps(experiment_data['result']),
                    experiment_data.get('cost_euros', 0.0),
                    experiment_data.get('shots', 1000),
                    experiment_data.get('notes', '')
                ))
                
                experiment_id = cur.fetchone()[0]
                self.conn.commit()
                
                logger.info(f"Stored experiment {experiment_id} in database")
                return experiment_id
                
        except Exception as e:
            logger.error(f"Failed to store experiment: {e}")
            self.conn.rollback()
            raise
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class SuperconductorResearcher:
    """Main research class that coordinates simulations and stores results."""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.pasqal_api_key = os.getenv('PASQAL_API_KEY')
        self.gcp_project_id = os.getenv('GCP_PROJECT_ID')
        
        # Configure simulation backends
        self.simulation_config = SimulationConfig(
            backend=BackendType.HYBRID,
            max_qubits_cpu=12,      # Use CPU for small systems
            max_qubits_gpu=25,      # Use GPU for medium systems  
            verbose=True
        )
        
        logger.info("SuperconductorResearcher initialized")
        if self.pasqal_api_key:
            logger.info("Pasqal Cloud API key found - cloud simulations enabled")
        else:
            logger.warning("No Pasqal API key - using local simulation only")
    
    def run_single_experiment(self, 
                            material_name: str,
                            temperature: float,
                            disorder_strength: float,
                            lattice_size: Tuple[int, int] = (2, 2),
                            use_tri_hybrid: bool = True,
                            notes: str = "") -> Dict[str, Any]:
        """
        Run a single tri-hybrid superconductor experiment combining analog, quantum, and classical approaches.
        
        Args:
            material_name: Name of material (e.g., 'YBCO', 'BSCCO')
            temperature: Temperature in Kelvin
            disorder_strength: Disorder strength (0.0 to 1.0)
            lattice_size: (nx, ny) lattice dimensions
            use_tri_hybrid: Whether to use full tri-hybrid approach or just analog
            notes: Optional notes about the experiment
            
        Returns:
            Dictionary with experiment results
        """
        start_time = time.time()
        
        try:
            # Get material from existing database
            material = get_material(material_name)
            logger.info(f"Running {'tri-hybrid' if use_tri_hybrid else 'analog-only'} experiment: {material_name} at {temperature}K with disorder {disorder_strength}")
            
            n_qubits = lattice_size[0] * lattice_size[1]
            
            if use_tri_hybrid:
                # Use the full tri-hybrid workflow
                config = WorkflowConfig(
                    temperature_range=(temperature - 10, temperature + 10),
                    n_temperatures=1,  # Single temperature
                    disorder_range=(disorder_strength, disorder_strength),
                    n_disorder_points=1,  # Single disorder value
                    lattice_size=lattice_size,
                    evolution_time=50.0,
                    n_shots=100,  # Reduced for faster execution
                    verbose=True
                )
                
                workflow = TriHybridWorkflow(material_name, config)
                
                # Run the complete tri-hybrid simulation
                discovery_result = workflow.discover_optimal_superconductor()
                
                # Extract results from tri-hybrid workflow
                analog_result = discovery_result.analog_results[0] if discovery_result.analog_results else None
                quantum_result = discovery_result.quantum_results[0] if discovery_result.quantum_results else None
                
                # Use classical transport analysis for Tc calculation
                transport = StrangeMetalTransport(material)
                if analog_result and quantum_result:
                    transport_analysis = transport.analyze_tri_hybrid_results([analog_result], [quantum_result])
                    tc_predicted = transport_analysis['optimal_conditions']['tc_suppressed']
                    confidence_score = 0.9  # High confidence for tri-hybrid
                    method_used = "tri_hybrid_full"
                elif analog_result:
                    # Fallback to analog-only
                    transport_analysis = transport.analyze_tri_hybrid_results([analog_result])
                    tc_predicted = transport_analysis['optimal_conditions']['tc_suppressed']
                    confidence_score = 0.7  # Lower confidence for analog-only
                    method_used = "analog_only_fallback"
                else:
                    raise RuntimeError("All tri-hybrid simulations failed")
                
                # Determine backend based on actual execution
                if analog_result and analog_result.get('cloud_job_id'):
                    backend = 'pasqal_cloud'
                    cost = 0.01 * analog_result.get('shots', 100)
                elif analog_result and analog_result.get('simulation_method') == 'real_pulser':
                    backend = 'local_simulation'
                    cost = 0.0
                else:
                    backend = 'local_gpu'
                    cost = 0.0
                
                analysis_result = {
                    'tc_predicted': float(tc_predicted),
                    'confidence_score': float(confidence_score),
                    'order_parameter': analog_result.get('order_parameter', 0.5) if analog_result else None,
                    'method_used': method_used,
                    'analog_backend': analog_result.get('simulation_method') if analog_result else None,
                    'quantum_circuit_depth': quantum_result.get('circuit_depth') if quantum_result else None,
                    'classical_enhancement': transport_analysis.get('max_tc_enhancement', 1.0),
                    'tri_hybrid_integration': True
                }
                
            else:
                # Fallback to analog-only simulation (original approach)
                disorder_pattern = create_realistic_cuprate_disorder(material_name, temperature=temperature)
                disorder_pattern.position_variance = disorder_strength * 0.5
                
                analog_sim = StrangeMetalAnalogSimulation(
                    material, 
                    disorder_pattern=disorder_pattern,
                    simulation_config=self.simulation_config
                )
                
                result = analog_sim.run_simulation(
                    temperature=temperature,
                    lattice_size=lattice_size,
                    evolution_time=50.0,
                    n_shots=100
                )
                
                if result is None or result.get('error'):
                    raise RuntimeError(f"Analog simulation failed: {result.get('error', 'Unknown error')}")
                
                # Use simplified Tc calculation for analog-only
                analysis_result = self._analyze_result(result, material, temperature, disorder_strength)
                analysis_result['tri_hybrid_integration'] = False
                
                simulation_method = result.get('simulation_method', 'unknown')
                if 'pasqal_cloud' in simulation_method.lower() and result.get('cloud_job_id'):
                    backend = 'pasqal_cloud'
                    cost = 0.01 * result.get('shots', 100)
                elif 'real_pulser' in simulation_method.lower():
                    backend = 'local_simulation'
                    cost = 0.0
                else:
                    backend = 'local_gpu'
                    cost = 0.0
            
            simulation_time = time.time() - start_time
            
            # Prepare experiment data for database
            experiment_data = {
                'material': material_name,
                'material_formula': getattr(material, 'chemical_formula', f'{material_name}_formula'),
                'temperature_k': temperature,
                'disorder_strength': disorder_strength,
                'n_qubits': n_qubits,
                'backend': backend,
                'simulation_time_seconds': simulation_time,
                'result': analysis_result,
                'cost_euros': cost,
                'shots': analysis_result.get('shots', 100),
                'notes': f"{'Tri-hybrid' if use_tri_hybrid else 'Analog-only'} | {notes}".strip()
            }
            
            # Store in database
            experiment_id = self.db.insert_experiment(experiment_data)
            
            logger.info(
                f"Experiment {experiment_id} completed: "
                f"Tc_predicted={analysis_result.get('tc_predicted', 'N/A')}K, "
                f"method={'tri-hybrid' if use_tri_hybrid else 'analog-only'}, "
                f"backend={backend}, time={simulation_time:.3f}s"
            )
            
            return {
                'experiment_id': experiment_id,
                'backend': backend,
                'simulation_time': simulation_time,
                'cost': cost,
                'tri_hybrid_used': use_tri_hybrid,
                **analysis_result
            }
            
        except Exception as e:
            simulation_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"Experiment failed after {simulation_time:.3f}s: {error_message}")
            
            # Store failed experiment in database with NULL result
            try:
                material = get_material(material_name)
                n_qubits = lattice_size[0] * lattice_size[1]
                
                experiment_data = {
                    'material': material_name,
                    'material_formula': getattr(material, 'chemical_formula', f'{material_name}_formula'),
                    'temperature_k': temperature,
                    'disorder_strength': disorder_strength,
                    'n_qubits': n_qubits,
                    'backend': 'failed_simulation',
                    'simulation_time_seconds': simulation_time,
                    'result': None,  # NULL for failed simulations
                    'cost_euros': 0.0,
                    'shots': 0,
                    'notes': f"FAILED: {error_message} | {notes}".strip()
                }
                
                experiment_id = self.db.insert_experiment(experiment_data)
                
                return {
                    'experiment_id': experiment_id,
                    'backend': 'failed_simulation',
                    'simulation_time': simulation_time,
                    'cost': 0.0,
                    'error': error_message,
                    'tc_predicted': None,
                    'confidence_score': None,
                    'order_parameter': None
                }
                
            except Exception as db_error:
                logger.error(f"Failed to store failed experiment: {db_error}")
                raise RuntimeError(f"Simulation failed: {error_message}") from e
    
    def _analyze_result(self, result: Dict[str, Any], material, temperature: float, disorder: float) -> Dict[str, Any]:
        """Analyze simulation result and extract key metrics."""
        
        # Get order parameter from simulation
        order_parameter = result.get('order_parameter', 0.5)
        
        # Estimate Tc based on material base Tc and disorder effects
        base_tc = material.tc_pristine
        
        # Simple model: disorder can enhance or suppress Tc
        # This is where the real physics research happens!
        disorder_factor = 1.0 + (disorder - 0.05) * 2.0  # Optimal around 5% disorder
        temperature_factor = np.exp(-(temperature - 100) / 200)  # Temperature dependence
        
        tc_predicted = base_tc * disorder_factor * (1 + order_parameter * 0.5) * temperature_factor
        
        # Calculate confidence based on order parameter and known physics
        confidence_score = order_parameter * 0.8 + 0.2  # Base confidence
        if tc_predicted > 140:  # Above current record
            confidence_score *= 0.9  # Slightly more skeptical of breakthrough results
        
        return {
            'tc_predicted': float(tc_predicted),
            'confidence_score': float(confidence_score),
            'order_parameter': float(order_parameter),
            'disorder_enhancement_factor': float(disorder_factor),
            'base_tc': float(base_tc),
            'simulation_metadata': {
                'backend_method': result.get('simulation_method', 'unknown'),
                'n_qubits': result.get('n_qubits', 0),
                'shots': result.get('shots', 0),
                'evolution_time': result.get('evolution_time', 0)
            }
        }
    
    def run_parameter_sweep(self, 
                          material_name: str,
                          temperature_range: Tuple[float, float] = (100, 300),
                          disorder_range: Tuple[float, float] = (0.0, 0.2),
                          n_temps: int = 5,
                          n_disorders: int = 5) -> List[Dict[str, Any]]:
        """Run a systematic parameter sweep."""
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_temps)
        disorders = np.linspace(disorder_range[0], disorder_range[1], n_disorders)
        
        results = []
        total_experiments = len(temperatures) * len(disorders)
        
        logger.info(f"Starting parameter sweep: {total_experiments} experiments")
        
        for i, temp in enumerate(temperatures):
            for j, disorder in enumerate(disorders):
                exp_num = i * len(disorders) + j + 1
                
                notes = f"Parameter sweep {exp_num}/{total_experiments}"
                
                try:
                    result = self.run_single_experiment(
                        material_name=material_name,
                        temperature=temp,
                        disorder_strength=disorder,
                        notes=notes
                    )
                    results.append(result)
                    
                    logger.info(f"Completed {exp_num}/{total_experiments}")
                    
                except Exception as e:
                    logger.error(f"Failed experiment {exp_num}: {e}")
                    continue
        
        logger.info(f"Parameter sweep completed: {len(results)}/{total_experiments} successful")
        return results
    
    def discover_promising_candidates(self, 
                                    materials: List[str] = None,
                                    target_tc: float = 140.0) -> List[Dict[str, Any]]:
        """Run focused experiments to find promising high-Tc candidates."""
        
        if materials is None:
            materials = ['YBCO', 'BSCCO', 'LSCO', 'FeSe']
        
        promising_results = []
        
        # Focus on potentially optimal conditions based on theory
        optimal_conditions = [
            (150, 0.05),   # Low disorder, moderate temperature
            (200, 0.08),   # Medium disorder, higher temperature  
            (120, 0.12),   # Higher disorder, lower temperature
        ]
        
        for material in materials:
            logger.info(f"Searching for promising conditions in {material}")
            
            for temp, disorder in optimal_conditions:
                try:
                    result = self.run_single_experiment(
                        material_name=material,
                        temperature=temp,
                        disorder_strength=disorder,
                        notes=f"Targeted discovery: {material} optimization"
                    )
                    
                    if result['tc_predicted'] > target_tc:
                        logger.info(f"🎉 PROMISING DISCOVERY: {material} Tc={result['tc_predicted']:.1f}K")
                        promising_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Discovery experiment failed: {e}")
                    continue
        
        return promising_results
    
    def close(self):
        """Clean up resources."""
        self.db.close()


def main():
    """Main research script entry point."""
    
    logger.info("🧪 Starting Superconductor Research Session")
    
    researcher = SuperconductorResearcher()
    
    try:
        # Example 1: Single experiment
        logger.info("Running single experiment...")
        result = researcher.run_single_experiment(
            material_name='YBCO',
            temperature=150.0,
            disorder_strength=0.08,
            notes="Initial test run"
        )
        print(f"Single experiment result: Tc = {result['tc_predicted']:.1f}K")
        
        # Example 2: Small parameter sweep
        logger.info("Running parameter sweep...")
        sweep_results = researcher.run_parameter_sweep(
            material_name='YBCO',
            temperature_range=(100, 200),
            disorder_range=(0.0, 0.15),
            n_temps=3,
            n_disorders=3
        )
        
        best_result = max(sweep_results, key=lambda x: x['tc_predicted'])
        print(f"Best from sweep: Tc = {best_result['tc_predicted']:.1f}K")
        
        # Example 3: Discovery search
        logger.info("Searching for promising discoveries...")
        discoveries = researcher.discover_promising_candidates(['YBCO', 'BSCCO'])
        
        if discoveries:
            print(f"Found {len(discoveries)} promising discoveries!")
            for disc in discoveries:
                print(f"  🌟 Tc = {disc['tc_predicted']:.1f}K (confidence: {disc['confidence_score']:.2f})")
        else:
            print("No breakthroughs found in this session - keep experimenting!")
        
    except KeyboardInterrupt:
        logger.info("Research session interrupted by user")
    except Exception as e:
        logger.error(f"Research session failed: {e}")
        raise
    finally:
        researcher.close()
        logger.info("Research session completed")


if __name__ == "__main__":
    main()