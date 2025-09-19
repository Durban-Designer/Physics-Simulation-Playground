"""
Tests for the materials module.
"""

import pytest
import numpy as np
from core.materials import (
    Material, CuprateMaterial, IronBasedMaterial, MaterialType,
    get_material, list_materials, create_custom_material,
    MATERIALS_DATABASE
)


class TestMaterial:
    """Test the base Material class."""
    
    def test_material_creation(self):
        """Test creating a basic material."""
        material = Material(
            name="TestMaterial",
            material_type=MaterialType.CUPRATE,
            lattice_constant=3.8,
            carrier_density=1e21,
            effective_mass=3.0,
            fermi_energy=0.5,
            tc_pristine=90.0,
            gap_symmetry="d",
            coherence_length=15.0,
            planckian_coefficient=1.0,
            strange_metal_range=(100, 300),
            resistivity_exponent=1.0
        )
        
        assert material.name == "TestMaterial"
        assert material.material_type == MaterialType.CUPRATE
        assert material.tc_pristine == 90.0
    
    def test_hopping_parameters(self):
        """Test hopping parameter calculation."""
        material = get_material("YBCO")
        hoppings = material.get_hopping_parameters()
        
        assert "t" in hoppings
        assert "t_prime" in hoppings
        assert "t_double_prime" in hoppings
        assert hoppings["t"] > 0
        assert hoppings["t_prime"] < 0  # Next-nearest neighbor is typically negative
    
    def test_interaction_strength(self):
        """Test interaction strength calculation."""
        material = get_material("YBCO")
        U = material.get_interaction_strength()
        
        assert U > 0
        assert isinstance(U, float)
    
    def test_strange_metal_regime(self):
        """Test strange metal temperature range."""
        material = get_material("YBCO")
        
        assert material.is_in_strange_metal_regime(200.0)  # Should be in range
        assert not material.is_in_strange_metal_regime(50.0)  # Below range
        assert not material.is_in_strange_metal_regime(400.0)  # Above range


class TestCuprateMaterial:
    """Test cuprate-specific functionality."""
    
    def test_cuprate_creation(self):
        """Test creating a cuprate material."""
        cuprate = CuprateMaterial(
            name="TestCuprate",
            material_type=MaterialType.CUPRATE,
            lattice_constant=3.8,
            carrier_density=1e21,
            effective_mass=3.0,
            fermi_energy=0.5,
            tc_pristine=90.0,
            gap_symmetry="d",
            coherence_length=15.0,
            planckian_coefficient=1.0,
            strange_metal_range=(100, 300),
            resistivity_exponent=1.0
        )
        
        assert cuprate.material_type == MaterialType.CUPRATE
        assert cuprate.gap_symmetry == "d"
        assert cuprate.optimal_doping == 0.16
    
    def test_oxygen_positions(self):
        """Test oxygen position generation."""
        cuprate = get_material("YBCO")
        positions = cuprate.get_oxygen_positions(3, 3)
        
        assert len(positions) == 18  # 3x3 = 9 Cu sites, 2 O per Cu = 18 O
        assert all(len(pos) == 2 for pos in positions)  # 2D positions
    
    def test_pseudogap_temperature(self):
        """Test pseudogap temperature calculation."""
        cuprate = get_material("YBCO")
        
        # Should be automatically calculated if not provided
        assert cuprate.pseudogap_temperature > cuprate.tc_pristine


class TestIronBasedMaterial:
    """Test iron-based superconductor functionality."""
    
    def test_iron_material_creation(self):
        """Test creating iron-based material."""
        iron_mat = get_material("FeSe")
        
        assert iron_mat.material_type == MaterialType.IRON_BASED
        assert iron_mat.gap_symmetry == "s+-"
        assert iron_mat.num_orbitals == 5
        assert len(iron_mat.orbital_energies) == 5
    
    def test_nematic_temperature(self):
        """Test nematic temperature calculation."""
        iron_mat = get_material("FeSe")
        
        # Nematic temperature should be close to or above SDW temperature
        assert iron_mat.nematic_temperature >= iron_mat.spin_density_wave_temp - 10


class TestMaterialDatabase:
    """Test the materials database functionality."""
    
    def test_get_material(self):
        """Test retrieving materials from database."""
        ybco = get_material("YBCO")
        assert ybco.name == "YBa2Cu3O7"
        assert isinstance(ybco, CuprateMaterial)
        
        fese = get_material("FeSe")
        assert fese.name == "FeSe"
        assert isinstance(fese, IronBasedMaterial)
    
    def test_get_nonexistent_material(self):
        """Test error handling for nonexistent materials."""
        with pytest.raises(ValueError):
            get_material("NonexistentMaterial")
    
    def test_list_materials(self):
        """Test listing available materials."""
        all_materials = list_materials()
        assert len(all_materials) > 0
        assert "YBCO" in all_materials
        assert "FeSe" in all_materials
        
        cuprates = list_materials(MaterialType.CUPRATE)
        assert "YBCO" in cuprates
        assert "BSCCO" in cuprates
        assert "FeSe" not in cuprates  # Should not include iron-based
        
        iron_based = list_materials(MaterialType.IRON_BASED)
        assert "FeSe" in iron_based
        assert "YBCO" not in iron_based  # Should not include cuprates
    
    def test_create_custom_material(self):
        """Test creating custom materials based on existing ones."""
        custom = create_custom_material(
            "CustomYBCO",
            "YBCO",
            tc_pristine=120.0,
            optimal_disorder=0.1
        )
        
        assert custom.name == "CustomYBCO"
        assert custom.tc_pristine == 120.0
        assert custom.optimal_disorder == 0.1
        assert isinstance(custom, CuprateMaterial)
        
        # Should inherit other properties from YBCO
        original = get_material("YBCO")
        assert custom.lattice_constant == original.lattice_constant


class TestMaterialPhysics:
    """Test physics calculations."""
    
    def test_strange_metal_properties(self):
        """Test strange metal property consistency."""
        for material_name in list_materials():
            material = get_material(material_name)
            
            # All materials should have T-linear resistivity in strange metal regime
            assert material.resistivity_exponent == pytest.approx(1.0, abs=0.2)
            
            # Planckian coefficient should be order 1
            assert 0.1 <= material.planckian_coefficient <= 10.0
            
            # Strange metal range should make sense
            T_min, T_max = material.strange_metal_range
            assert T_min > 0
            assert T_max > T_min
            assert T_max <= 500  # Reasonable upper bound
    
    def test_material_consistency(self):
        """Test internal consistency of material parameters."""
        for material_name in list_materials():
            material = get_material(material_name)
            
            # Basic sanity checks
            assert material.lattice_constant > 0
            assert material.carrier_density > 0
            assert material.effective_mass > 0
            assert material.fermi_energy > 0
            assert material.tc_pristine > 0
            assert material.coherence_length > 0
            
            # Coherence length should be reasonable (in Angstroms)
            assert 1.0 <= material.coherence_length <= 1000.0  # 1 Å to 1000 Å
            
            # Gap symmetry should be valid
            assert material.gap_symmetry in ["s", "d", "p", "s+-", "s++"]


if __name__ == "__main__":
    pytest.main([__file__])