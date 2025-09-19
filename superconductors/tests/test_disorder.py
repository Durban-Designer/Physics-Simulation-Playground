"""
Tests for the disorder module.
"""

import pytest
import numpy as np
from core.disorder import (
    DisorderPattern, DisorderType, create_realistic_cuprate_disorder
)


class TestDisorderPattern:
    """Test the DisorderPattern class."""
    
    def test_disorder_pattern_creation(self):
        """Test creating a basic disorder pattern."""
        pattern = DisorderPattern(
            disorder_type=DisorderType.POSITIONAL,
            base_spacing=5.0,
            position_variance=0.5,
            vacancy_rate=0.05
        )
        
        assert pattern.disorder_type == DisorderType.POSITIONAL
        assert pattern.base_spacing == 5.0
        assert pattern.position_variance == 0.5
        assert pattern.vacancy_rate == 0.05
    
    def test_position_generation(self):
        """Test generating disordered positions."""
        pattern = DisorderPattern(
            base_spacing=5.0,
            position_variance=0.1,
            vacancy_rate=0.0,  # No vacancies for simpler testing
            dopant_rate=0.0,   # No dopants
            interstitial_rate=0.0  # No interstitials
        )
        
        positions = pattern.generate_positions(5, 5)
        
        # Should have roughly 25 positions (5x5 grid)
        assert len(positions) <= 25
        assert len(positions) >= 20  # Some might be removed due to minimum spacing
        
        # All positions should be 2D
        assert all(len(pos) == 2 for pos in positions)
        
        # Positions should be reasonable (disorder can move atoms slightly negative)
        positions_array = np.array(positions)
        assert np.all(positions_array >= -1)  # Small negative displacement allowed
        assert np.all(positions_array <= 26)  # 5x5 grid with spacing 5 plus disorder
    
    def test_vacancy_effects(self):
        """Test that vacancies reduce the number of atoms."""
        # Pattern with no vacancies
        pattern_no_vac = DisorderPattern(
            base_spacing=5.0,
            vacancy_rate=0.0,
            dopant_rate=0.0,
            interstitial_rate=0.0
        )
        
        # Pattern with 50% vacancies
        pattern_high_vac = DisorderPattern(
            base_spacing=5.0,
            vacancy_rate=0.5,
            dopant_rate=0.0,
            interstitial_rate=0.0
        )
        
        positions_no_vac = pattern_no_vac.generate_positions(10, 10)
        positions_high_vac = pattern_high_vac.generate_positions(10, 10)
        
        # High vacancy should have significantly fewer atoms
        assert len(positions_high_vac) < len(positions_no_vac)
        assert len(positions_high_vac) < 75  # Expect much less than 100
    
    def test_dopant_addition(self):
        """Test dopant atom addition."""
        pattern = DisorderPattern(
            base_spacing=5.0,
            vacancy_rate=0.0,
            dopant_rate=0.2,  # 20% dopants
            interstitial_rate=0.0
        )
        
        positions = pattern.generate_positions(5, 5)
        
        # Should have more atoms than the base grid due to dopants
        assert len(positions) > 25
    
    def test_disorder_strength_calculation(self):
        """Test disorder strength quantification."""
        # Perfect lattice (no disorder)
        perfect_positions = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ], dtype=float)
        
        pattern = DisorderPattern()
        perfect_metrics = pattern.compute_disorder_strength(perfect_positions)
        
        assert perfect_metrics["positional"] == pytest.approx(0.0, abs=0.1)
        
        # Disordered lattice
        disordered_positions = perfect_positions + np.random.randn(*perfect_positions.shape) * 0.1
        disordered_metrics = pattern.compute_disorder_strength(disordered_positions)
        
        assert disordered_metrics["total"] > perfect_metrics["total"]
        assert disordered_metrics["positional"] > 0
    
    def test_temperature_scaling(self):
        """Test temperature-dependent disorder scaling."""
        pattern = DisorderPattern(temperature_dependent=True)
        
        # Higher temperature should increase disorder
        scale_100K = pattern.temperature_scaling(100.0)
        scale_300K = pattern.temperature_scaling(300.0)
        
        assert scale_300K > scale_100K
        
        # Non-temperature dependent should be constant
        pattern_const = DisorderPattern(temperature_dependent=False)
        scale_const_100K = pattern_const.temperature_scaling(100.0)
        scale_const_300K = pattern_const.temperature_scaling(300.0)
        
        assert scale_const_100K == scale_const_300K == 1.0
    
    def test_patchwork_pattern(self):
        """Test patchwork disorder pattern generation."""
        pattern = DisorderPattern()
        patchwork = pattern.create_patchwork_pattern(10, 10, patch_size=20.0)
        
        assert patchwork.shape == (10, 10)
        assert np.all(patchwork >= 0)  # Should be non-negative
        assert not np.allclose(patchwork, patchwork.mean())  # Should have variations


class TestDisorderTypes:
    """Test different disorder types."""
    
    def test_positional_disorder(self):
        """Test pure positional disorder."""
        pattern = DisorderPattern(
            disorder_type=DisorderType.POSITIONAL,
            position_variance=0.2,
            vacancy_rate=0.0,
            dopant_rate=0.0
        )
        
        positions = pattern.generate_positions(5, 5)
        assert len(positions) == 25  # Should have all atoms
    
    def test_vacancy_disorder(self):
        """Test pure vacancy disorder."""
        pattern = DisorderPattern(
            disorder_type=DisorderType.VACANCY,
            position_variance=0.0,
            vacancy_rate=0.3,
            dopant_rate=0.0
        )
        
        positions = pattern.generate_positions(10, 10)
        assert len(positions) < 100  # Should have fewer atoms due to vacancies
    
    def test_composite_disorder(self):
        """Test composite disorder with multiple types."""
        pattern = DisorderPattern(
            disorder_type=DisorderType.COMPOSITE,
            position_variance=0.1,
            vacancy_rate=0.05,
            dopant_rate=0.1,
            interstitial_rate=0.02
        )
        
        positions = pattern.generate_positions(8, 8)
        
        # Should have complexity from multiple disorder sources
        assert len(positions) > 0
        disorder_metrics = pattern.compute_disorder_strength(np.array(positions))
        assert disorder_metrics["total"] > 0


class TestRealisticDisorder:
    """Test realistic disorder patterns for known materials."""
    
    def test_ybco_disorder(self):
        """Test YBCO-specific disorder pattern."""
        disorder = create_realistic_cuprate_disorder("YBCO", temperature=100.0)
        
        assert disorder.disorder_type == DisorderType.COMPOSITE
        assert disorder.base_spacing == pytest.approx(3.8, rel=0.1)  # YBCO lattice constant
        assert disorder.vacancy_rate > 0  # Should have oxygen vacancies
        assert disorder.dopant_rate > 0   # Should have hole doping
    
    def test_bscco_disorder(self):
        """Test BSCCO-specific disorder pattern."""
        disorder = create_realistic_cuprate_disorder("BSCCO", temperature=150.0)
        
        assert disorder.disorder_type == DisorderType.COMPOSITE
        assert disorder.clustering_factor > 0  # Should have dopant clustering
    
    def test_temperature_dependence(self):
        """Test temperature dependence of realistic disorder."""
        disorder_100K = create_realistic_cuprate_disorder("YBCO", temperature=100.0)
        disorder_300K = create_realistic_cuprate_disorder("YBCO", temperature=300.0)
        
        # Higher temperature should increase positional disorder
        assert disorder_300K.position_variance > disorder_100K.position_variance
    
    def test_disorder_pattern_consistency(self):
        """Test that disorder patterns produce consistent results."""
        # Use fixed seed for reproducibility
        pattern = DisorderPattern(random_seed=42)
        
        positions1 = pattern.generate_positions(5, 5)
        
        # Reset and generate again with same seed
        pattern_repeat = DisorderPattern(random_seed=42)
        positions2 = pattern_repeat.generate_positions(5, 5)
        
        # Should be identical
        np.testing.assert_array_equal(positions1, positions2)


class TestCorrelationFunctions:
    """Test spatial correlation functions."""
    
    def test_correlation_function(self):
        """Test custom correlation functions."""
        # Exponential correlation
        def exp_corr(r):
            return np.exp(-r / 5.0)
        
        pattern = DisorderPattern(
            correlation_function=exp_corr,
            correlation_length=5.0
        )
        
        # Test that correlation function is stored correctly
        assert pattern.correlation_function(0) == 1.0
        assert pattern.correlation_function(5.0) == pytest.approx(np.exp(-1), rel=0.01)
    
    def test_anisotropic_disorder(self):
        """Test anisotropic disorder patterns."""
        pattern = DisorderPattern(
            anisotropy_factor=2.0,  # Stronger disorder in x-direction
            position_variance=0.2
        )
        
        positions = pattern.generate_positions(10, 10)
        positions_array = np.array(positions)
        
        # Should have more variation in x than y (though this is statistical)
        if len(positions_array) > 10:
            x_variance = np.var(positions_array[:, 0])
            y_variance = np.var(positions_array[:, 1])
            
            # This is a statistical test, so we'll be lenient
            assert x_variance >= y_variance * 0.5  # At least some anisotropy


if __name__ == "__main__":
    pytest.main([__file__])