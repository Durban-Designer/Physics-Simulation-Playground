"""
Disorder patterns for strange metal simulations.

This module defines various disorder patterns that are crucial for engineering
strange metal behavior and enhancing superconductivity according to the Patel
mechanism.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal


class DisorderType(Enum):
    """Types of disorder in the system."""
    POSITIONAL = "positional"  # Atomic position variations
    VACANCY = "vacancy"  # Missing atoms
    DOPANT = "dopant"  # Substitutional atoms
    INTERSTITIAL = "interstitial"  # Extra atoms between lattice sites
    CHARGE = "charge"  # Local charge variations
    MAGNETIC = "magnetic"  # Local magnetic moments
    COMPOSITE = "composite"  # Multiple disorder types


@dataclass
class DisorderPattern:
    """
    Defines atomic disorder patterns in strange metals.
    
    The interplay between disorder and quantum entanglement is key to
    producing T-linear resistivity in strange metals.
    """
    
    # Basic disorder parameters
    disorder_type: DisorderType = DisorderType.COMPOSITE
    base_spacing: float = 5.0  # μm (Pasqal scale) or Angstroms (real scale)
    
    # Positional disorder
    position_variance: float = 0.5  # Standard deviation of position fluctuations
    correlation_length: float = 10.0  # Spatial correlation of disorder
    
    # Point defects
    vacancy_rate: float = 0.05  # Fraction of missing atoms
    dopant_rate: float = 0.10  # Fraction of dopant atoms
    interstitial_rate: float = 0.01  # Fraction of interstitial defects
    
    # Dopant clustering
    dopant_positions: Optional[List[Tuple[float, float]]] = None
    clustering_factor: float = 0.2  # Probability of dopant clustering
    cluster_size: int = 3  # Average cluster size
    
    # Charge/magnetic disorder
    charge_modulation_amplitude: float = 0.1  # Local charge variations
    magnetic_disorder_strength: float = 0.0  # Random local moments
    
    # Advanced disorder features
    correlation_function: Optional[Callable] = None  # Spatial correlations
    anisotropy_factor: float = 1.0  # x/y disorder ratio
    temperature_dependent: bool = True  # Disorder increases with T
    
    # Disorder realization seed
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize random number generator and correlation function."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if self.correlation_function is None:
            # Default exponential correlation
            self.correlation_function = lambda r: np.exp(-r / self.correlation_length)
    
    def generate_positions(self, nx: int, ny: int, 
                          perfect_lattice: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate disordered atomic positions.
        
        Args:
            nx, ny: Number of sites in x and y directions
            perfect_lattice: Optional starting lattice positions
            
        Returns:
            Array of (x, y) positions with disorder
        """
        if perfect_lattice is None:
            # Create perfect square lattice
            x = np.arange(nx) * self.base_spacing
            y = np.arange(ny) * self.base_spacing
            xx, yy = np.meshgrid(x, y, indexing='ij')
            perfect_lattice = np.column_stack([xx.ravel(), yy.ravel()])
        
        n_sites = len(perfect_lattice)
        positions = []
        
        # Generate correlated disorder field
        disorder_field = self._generate_correlated_disorder(nx, ny)
        
        # Apply different types of disorder
        site_index = 0
        for i in range(nx):
            for j in range(ny):
                base_pos = perfect_lattice[site_index]
                
                # Skip vacancies
                if np.random.random() < self.vacancy_rate:
                    site_index += 1
                    continue
                
                # Positional disorder with spatial correlation
                if self.disorder_type in [DisorderType.POSITIONAL, DisorderType.COMPOSITE]:
                    dx = disorder_field[i, j, 0] * self.position_variance
                    dy = disorder_field[i, j, 1] * self.position_variance
                    
                    # Apply anisotropy
                    dx *= self.anisotropy_factor
                    
                    pos = base_pos + np.array([dx, dy])
                else:
                    pos = base_pos
                
                positions.append(pos)
                site_index += 1
        
        # Add dopants
        if self.dopant_rate > 0:
            positions.extend(self._add_dopants(nx, ny))
        
        # Add interstitials
        if self.interstitial_rate > 0:
            positions.extend(self._add_interstitials(perfect_lattice))
        
        return np.array(positions)
    
    def _generate_correlated_disorder(self, nx: int, ny: int) -> np.ndarray:
        """Generate spatially correlated random disorder field."""
        # Create correlation matrix based on distance
        x = np.arange(nx)
        y = np.arange(ny)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Compute correlation matrix
        dist_matrix = distance_matrix(points, points)
        correlation_matrix = self.correlation_function(dist_matrix)
        
        # Add small diagonal term for numerical stability
        correlation_matrix += 1e-6 * np.eye(len(points))
        
        # Generate correlated random field
        mean = np.zeros(len(points))
        disorder_x = multivariate_normal.rvs(mean, correlation_matrix)
        disorder_y = multivariate_normal.rvs(mean, correlation_matrix)
        
        # Reshape to grid
        field = np.zeros((nx, ny, 2))
        field[:, :, 0] = disorder_x.reshape(nx, ny)
        field[:, :, 1] = disorder_y.reshape(nx, ny)
        
        return field
    
    def _add_dopants(self, nx: int, ny: int) -> List[Tuple[float, float]]:
        """Add dopant atoms with optional clustering."""
        dopant_positions = []
        n_dopants = int(nx * ny * self.dopant_rate)
        
        if self.dopant_positions is not None:
            # Use specified dopant positions
            base_positions = self.dopant_positions[:n_dopants]
        else:
            # Random dopant positions
            base_positions = []
            for _ in range(n_dopants):
                x = np.random.uniform(0, nx * self.base_spacing)
                y = np.random.uniform(0, ny * self.base_spacing)
                base_positions.append((x, y))
        
        # Add clustering
        for base_x, base_y in base_positions:
            if np.random.random() < self.clustering_factor:
                # Create cluster around this dopant
                n_cluster = np.random.poisson(self.cluster_size)
                for _ in range(n_cluster):
                    # Gaussian distribution around center
                    dx = np.random.normal(0, self.base_spacing / 4)
                    dy = np.random.normal(0, self.base_spacing / 4)
                    dopant_positions.append((base_x + dx, base_y + dy))
            else:
                dopant_positions.append((base_x, base_y))
        
        return dopant_positions
    
    def _add_interstitials(self, lattice_positions: np.ndarray) -> List[Tuple[float, float]]:
        """Add interstitial defects between lattice sites."""
        interstitials = []
        n_interstitials = int(len(lattice_positions) * self.interstitial_rate)
        
        for _ in range(n_interstitials):
            # Pick two random lattice sites
            idx1, idx2 = np.random.choice(len(lattice_positions), 2, replace=False)
            pos1, pos2 = lattice_positions[idx1], lattice_positions[idx2]
            
            # Place interstitial between them with some randomness
            interstitial_pos = (pos1 + pos2) / 2 + np.random.randn(2) * self.base_spacing * 0.1
            interstitials.append(tuple(interstitial_pos))
        
        return interstitials
    
    def compute_disorder_strength(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Quantify the disorder strength from a set of positions.
        
        Returns various disorder metrics that affect strange metal behavior.
        """
        if len(positions) < 2:
            return {"total": 0.0, "positional": 0.0, "density": 0.0}
        
        # Compute nearest neighbor distances
        dist_matrix = distance_matrix(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_distances = np.min(dist_matrix, axis=1)
        
        # Positional disorder: variance in nearest neighbor distances
        positional_disorder = np.std(nn_distances) / np.mean(nn_distances)
        
        # Density fluctuations: local density variations
        density_disorder = self._compute_density_fluctuations(positions)
        
        # Pair correlation function deviation from perfect crystal
        pair_correlation_disorder = self._compute_pair_correlation_deviation(positions)
        
        # Total disorder strength (combines all effects)
        total_disorder = np.sqrt(
            positional_disorder**2 + 
            density_disorder**2 + 
            pair_correlation_disorder**2
        )
        
        return {
            "total": total_disorder,
            "positional": positional_disorder,
            "density": density_disorder,
            "pair_correlation": pair_correlation_disorder,
            "mean_nn_distance": np.mean(nn_distances),
            "nn_distance_std": np.std(nn_distances)
        }
    
    def _compute_density_fluctuations(self, positions: np.ndarray, 
                                     n_bins: int = 10) -> float:
        """Compute local density fluctuations."""
        # Divide space into bins and count atoms
        x_bins = np.linspace(positions[:, 0].min(), positions[:, 0].max(), n_bins)
        y_bins = np.linspace(positions[:, 1].min(), positions[:, 1].max(), n_bins)
        
        counts, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], 
                                      bins=[x_bins, y_bins])
        
        # Normalize by expected count
        expected_count = len(positions) / (n_bins - 1)**2
        density_fluctuation = np.std(counts.ravel()) / expected_count
        
        return density_fluctuation
    
    def _compute_pair_correlation_deviation(self, positions: np.ndarray, 
                                           max_distance: Optional[float] = None) -> float:
        """Compute deviation of pair correlation from perfect crystal."""
        if max_distance is None:
            max_distance = 5 * self.base_spacing
        
        # Compute radial distribution function
        dist_matrix = distance_matrix(positions, positions)
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        
        # Histogram of distances
        bins = np.linspace(0, max_distance, 50)
        hist, bin_edges = np.histogram(distances, bins=bins, density=True)
        
        # Expected peaks for perfect crystal at multiples of base_spacing
        expected_peaks = np.arange(1, 6) * self.base_spacing
        
        # Compute deviation from expected peak positions
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        deviation = 0.0
        
        for peak in expected_peaks:
            if peak < max_distance:
                # Find actual peak near expected position
                idx_range = np.where(np.abs(bin_centers - peak) < self.base_spacing/2)[0]
                if len(idx_range) > 0:
                    actual_peak_idx = idx_range[np.argmax(hist[idx_range])]
                    actual_peak = bin_centers[actual_peak_idx]
                    deviation += (actual_peak - peak)**2 / peak**2
        
        return np.sqrt(deviation)
    
    def temperature_scaling(self, temperature: float, reference_temp: float = 300.0) -> float:
        """
        Scale disorder strength with temperature.
        
        Higher temperatures increase disorder through thermal vibrations.
        """
        if not self.temperature_dependent:
            return 1.0
        
        # Debye-Waller factor approximation
        return np.sqrt(temperature / reference_temp)
    
    def create_patchwork_pattern(self, nx: int, ny: int, 
                                patch_size: float = 20.0) -> np.ndarray:
        """
        Create the 'patchwork' disorder pattern characteristic of strange metals.
        
        This creates regions with different local disorder strengths,
        which is key to the strange metal mechanism.
        """
        x = np.arange(nx) * self.base_spacing
        y = np.arange(ny) * self.base_spacing
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # Create multiple disorder patches
        n_patches = int((nx * ny * self.base_spacing**2) / patch_size**2)
        disorder_map = np.zeros((nx, ny))
        
        for _ in range(n_patches):
            # Random patch center
            center_x = np.random.uniform(0, nx * self.base_spacing)
            center_y = np.random.uniform(0, ny * self.base_spacing)
            
            # Random patch strength
            strength = np.random.uniform(0.5, 1.5)
            
            # Gaussian patch
            distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            patch = strength * np.exp(-distances**2 / (2 * patch_size**2))
            
            disorder_map += patch
        
        # Normalize
        disorder_map /= disorder_map.mean()
        
        return disorder_map


def create_realistic_cuprate_disorder(material_name: str = "YBCO",
                                     temperature: float = 100.0) -> DisorderPattern:
    """
    Create a realistic disorder pattern for cuprate superconductors.
    
    Based on experimental observations of oxygen vacancies, dopant clustering,
    and thermal vibrations.
    """
    # Temperature-dependent parameters
    position_variance = 0.05 * np.sqrt(temperature / 100.0)  # Angstroms
    
    # Material-specific parameters
    if material_name == "YBCO":
        vacancy_rate = 0.03  # 3% oxygen vacancies
        dopant_rate = 0.16  # Optimal doping
        clustering_factor = 0.3  # Moderate clustering
    elif material_name == "BSCCO":
        vacancy_rate = 0.05  # More disorder
        dopant_rate = 0.15
        clustering_factor = 0.4  # Stronger clustering
    else:
        vacancy_rate = 0.02
        dopant_rate = 0.15
        clustering_factor = 0.2
    
    return DisorderPattern(
        disorder_type=DisorderType.COMPOSITE,
        base_spacing=3.8,  # Angstroms
        position_variance=position_variance,
        correlation_length=15.0,  # Few unit cells
        vacancy_rate=vacancy_rate,
        dopant_rate=dopant_rate,
        clustering_factor=clustering_factor,
        cluster_size=4,
        charge_modulation_amplitude=0.1,
        temperature_dependent=True
    )