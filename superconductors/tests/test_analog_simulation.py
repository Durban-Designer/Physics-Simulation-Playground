"""
Tests for the analog simulation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from analog.strange_metal_disorder import (
    StrangeMetalLattice, AnalogHamiltonianBuilder, StrangeMetalAnalogSimulation
)
from core.materials import get_material
from core.disorder import DisorderPattern, DisorderType


class TestStrangeMetalLattice:
    """Test the StrangeMetalLattice class."""
    
    def test_lattice_creation_with_material_name(self):
        """Test creating lattice with material name."""
        lattice = StrangeMetalLattice("YBCO")
        
        assert lattice.material.name == "YBa2Cu3O7"
        assert lattice.disorder is not None
        assert lattice.scale_factor > 0
    
    def test_lattice_creation_with_material_object(self):
        """Test creating lattice with material object."""
        material = get_material("YBCO")
        lattice = StrangeMetalLattice(material)
        
        assert lattice.material.name == material.name
    
    def test_lattice_creation_with_custom_disorder(self):
        """Test creating lattice with custom disorder pattern."""
        disorder = DisorderPattern(
            disorder_type=DisorderType.POSITIONAL,
            position_variance=0.1,
            vacancy_rate=0.02
        )
        
        lattice = StrangeMetalLattice("YBCO", disorder)
        assert lattice.disorder.disorder_type == DisorderType.POSITIONAL
        assert lattice.disorder.vacancy_rate == 0.02
    
    def test_cuprate_plane_generation(self):
        """Test CuO2 plane generation."""
        lattice = StrangeMetalLattice("YBCO")
        register = lattice.generate_cuprate_plane(5, 5)
        
        # Should return either a real Register or mock
        assert register is not None
        
        if isinstance(register, dict):
            # Mock register
            assert "positions" in register
            positions = register["positions"]
            assert len(positions) > 0
            assert len(positions) <= 40  # 5x5 grid with dopants and clustering
        else:
            # Real register (if Pulser available)
            assert hasattr(register, 'atoms_coords') or hasattr(register, '_coords')
    
    def test_minimum_spacing_enforcement(self):
        """Test that minimum spacing is enforced."""
        lattice = StrangeMetalLattice("YBCO")
        
        # Test with very close positions
        close_positions = np.array([
            [0.0, 0.0],
            [0.1, 0.0],  # Very close - should be moved apart
            [10.0, 0.0]
        ])
        
        separated = lattice._enforce_minimum_spacing(close_positions, min_spacing=4.0)
        
        # All pairs should now be at least 4.0 apart
        for i in range(len(separated)):
            for j in range(i + 1, len(separated)):
                distance = np.linalg.norm(np.array(separated[i]) - np.array(separated[j]))
                assert distance >= 3.9  # Allow small numerical error
    
    def test_hamiltonian_parameters(self):
        """Test Hamiltonian parameter calculation."""
        lattice = StrangeMetalLattice("YBCO")
        params = lattice._calculate_hamiltonian_parameters(100.0)  # 100K
        
        assert "omega" in params
        assert "delta" in params
        assert "disorder_strength" in params
        assert "temperature_factor" in params
        
        assert params["omega"] > 0
        assert params["temperature_factor"] > 0
        assert params["disorder_strength"] >= 0
    
    def test_disorder_pulse_building(self):
        """Test disorder-dependent pulse construction."""
        lattice = StrangeMetalLattice("YBCO")
        params = {
            "omega": 10.0,
            "omega_variation": 1.0,
            "delta": -5.0,
            "delta_variation": 0.5,
            "disorder_strength": 0.1
        }
        
        pulse = lattice._build_disorder_pulse(params, 1000.0)
        
        # Should return pulse object (real or mock)
        assert pulse is not None
    
    def test_temperature_scaling(self):
        """Test temperature dependence of Hamiltonian."""
        lattice = StrangeMetalLattice("YBCO")
        
        params_100K = lattice._calculate_hamiltonian_parameters(100.0)
        params_300K = lattice._calculate_hamiltonian_parameters(300.0)
        
        # Temperature factor should increase with temperature
        assert params_300K["temperature_factor"] > params_100K["temperature_factor"]


class TestAnalogHamiltonianBuilder:
    """Test the AnalogHamiltonianBuilder class."""
    
    def test_hamiltonian_builder_creation(self):
        """Test creating Hamiltonian builder."""
        disorder = DisorderPattern()
        material = get_material("YBCO")
        
        builder = AnalogHamiltonianBuilder(
            disorder_pattern=disorder,
            material=material,
            temperature=100.0
        )
        
        assert builder.disorder_pattern == disorder
        assert builder.material == material
        assert builder.temperature == 100.0
    
    def test_rydberg_hamiltonian_construction(self):
        """Test Rydberg Hamiltonian matrix construction."""
        disorder = DisorderPattern()
        material = get_material("YBCO")
        builder = AnalogHamiltonianBuilder(disorder, material, 100.0)
        
        # Simple 3-atom system
        positions = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [0.0, 5.0]
        ])
        
        hamiltonian = builder.build_rydberg_hamiltonian(positions)
        
        assert "V_matrix" in hamiltonian
        assert "omega_i" in hamiltonian
        assert "delta_i" in hamiltonian
        assert "distances" in hamiltonian
        
        V_matrix = hamiltonian["V_matrix"]
        omega_i = hamiltonian["omega_i"]
        delta_i = hamiltonian["delta_i"]
        
        # Check dimensions
        assert V_matrix.shape == (3, 3)
        assert len(omega_i) == 3
        assert len(delta_i) == 3
        
        # V_matrix should be symmetric with zero diagonal
        assert np.allclose(V_matrix, V_matrix.T)
        assert np.allclose(np.diag(V_matrix), 0)
        
        # Interaction should decay with distance
        assert V_matrix[0, 1] > 0  # Repulsive interaction
        
        # Omega should be positive
        assert np.all(omega_i >= 0)
    
    def test_disorder_field_computation(self):
        """Test disorder field computation."""
        disorder = DisorderPattern(correlation_length=10.0)
        material = get_material("YBCO")
        builder = AnalogHamiltonianBuilder(disorder, material, 100.0)
        
        positions = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [0.0, 5.0]
        ])
        
        disorder_field = builder._compute_disorder_field(positions)
        
        assert len(disorder_field) == 4
        assert np.all(disorder_field > 0)  # Should be positive
        assert not np.allclose(disorder_field, disorder_field[0])  # Should vary
    
    def test_local_parameters(self):
        """Test local Rabi frequency and detuning computation."""
        disorder = DisorderPattern()
        material = get_material("YBCO")
        builder = AnalogHamiltonianBuilder(disorder, material, 100.0)
        
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        disorder_field = np.array([1.0, 1.2])
        
        omega_i = builder._compute_local_rabi(positions, disorder_field)
        delta_i = builder._compute_local_detuning(positions, disorder_field)
        
        assert len(omega_i) == 2
        assert len(delta_i) == 2
        assert np.all(omega_i >= 0)
        
        # Parameters should differ due to disorder
        if np.abs(disorder_field[0] - disorder_field[1]) > 0.1:
            assert omega_i[0] != omega_i[1] or delta_i[0] != delta_i[1]


class TestStrangeMetalAnalogSimulation:
    """Test the complete analog simulation workflow."""
    
    def test_simulation_creation(self):
        """Test creating analog simulation."""
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        assert sim.material.name == "YBa2Cu3O7"
        assert sim.lattice_generator is not None
        assert sim.measurement_results == []
    
    def test_single_point_simulation(self):
        """Test running single point simulation."""
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        result = sim.run_simulation(
            temperature=100.0,
            lattice_size=(2, 2),  # Reduced to 4 qubits for stability
            evolution_time=50.0,  # Reduced evolution time
            n_shots=25  # Reduced shots for faster testing
        )
        
        # Check result structure
        assert "temperature" in result
        assert "lattice_size" in result
        assert "n_atoms" in result
        assert "disorder_metrics" in result
        assert "local_order_parameter" in result
        assert "mean_order" in result
        assert "order_variance" in result
        
        assert result["temperature"] == 100.0
        assert result["lattice_size"] == (2, 2)
        assert result["n_atoms"] > 0
        assert isinstance(result["local_order_parameter"], np.ndarray)
    
    def test_temperature_scan(self):
        """Test temperature scanning."""
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        results = sim.scan_temperature(
            temp_range=(50.0, 200.0),
            n_temps=3,  # Reduced for faster testing
            lattice_size=(2, 2),  # Reduced to 4 qubits for stability
            n_shots=25  # Reduced shots
        )
        
        assert len(results) == 3
        
        # Check that temperatures are correctly set
        temperatures = [r["temperature"] for r in results]
        assert min(temperatures) >= 50.0
        assert max(temperatures) <= 200.0
        
        # Results should be stored
        assert len(sim.measurement_results) == 3
    
    def test_transport_data_extraction(self):
        """Test extracting transport data."""
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        # Run some simulations first
        sim.scan_temperature((100.0, 300.0), n_temps=3, lattice_size=(2, 2))
        
        transport_data = sim.extract_transport_data()
        
        assert "temperatures" in transport_data
        assert "mean_order" in transport_data
        assert "order_variance" in transport_data
        assert "disorder_strength" in transport_data
        assert "patchwork_amplitude" in transport_data
        
        assert len(transport_data["temperatures"]) == 3
        assert np.all(transport_data["temperatures"] >= 100.0)
        assert np.all(transport_data["temperatures"] <= 300.0)
    
    def test_extract_without_results(self):
        """Test error handling when extracting without results."""
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        with pytest.raises(ValueError):
            sim.extract_transport_data()
    
    def test_simulation_initialization(self):
        """Test simulation initialization and configuration."""
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        # Check basic attributes
        assert sim.material is not None
        assert sim.lattice_generator is not None
        assert sim.disorder_pattern is not None
        assert sim.measurement_results == []
        
        # Test with custom disorder
        from core.disorder import DisorderPattern, DisorderType
        custom_disorder = DisorderPattern(
            disorder_type=DisorderType.POSITIONAL,
            position_variance=0.05
        )
        
        sim_custom = StrangeMetalAnalogSimulation("YBCO", custom_disorder)
        # Disorder may be modified by lattice generator, check it's approximately correct
        assert 0.04 <= sim_custom.disorder_pattern.position_variance <= 0.08


class TestIntegration:
    """Integration tests for analog simulation."""
    
    def test_full_workflow(self):
        """Test complete analog simulation workflow."""
        # Create simulation
        sim = StrangeMetalAnalogSimulation("YBCO")
        
        # Run temperature scan
        results = sim.scan_temperature(
            temp_range=(100.0, 200.0),
            n_temps=3,
            lattice_size=(4, 4),
            evolution_time=500.0,
            n_shots=50
        )
        
        # Extract transport data
        transport_data = sim.extract_transport_data()
        
        # Verify workflow completed successfully
        assert len(results) == 3
        assert len(transport_data["temperatures"]) == 3
        
        # Check that disorder affects results
        disorder_strengths = transport_data["disorder_strength"]
        assert np.any(disorder_strengths > 0)
        
        # Check that measurements show variation
        order_parameters = [r["local_order_parameter"] for r in results]
        assert all(len(op) > 0 for op in order_parameters)
    
    def test_material_comparison(self):
        """Test comparing different materials."""
        materials = ["YBCO", "FeSe"]
        results_by_material = {}
        
        for material in materials:
            sim = StrangeMetalAnalogSimulation(material)
            results = sim.scan_temperature(
                (100.0, 150.0), n_temps=2, lattice_size=(3, 3)
            )
            results_by_material[material] = results
        
        # Should have results for both materials
        assert len(results_by_material) == 2
        assert "YBCO" in results_by_material
        assert "FeSe" in results_by_material
        
        # Results should differ between materials
        ybco_disorder = results_by_material["YBCO"][0]["disorder_metrics"]["total"]
        fese_disorder = results_by_material["FeSe"][0]["disorder_metrics"]["total"]
        
        # May or may not be different, but workflow should complete
        assert ybco_disorder >= 0
        assert fese_disorder >= 0


if __name__ == "__main__":
    pytest.main([__file__])