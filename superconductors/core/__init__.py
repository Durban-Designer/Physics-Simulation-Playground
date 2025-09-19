"""
Core physics modules for the Tri-Hybrid Superconductor Discovery Platform.
"""

from .materials import Material, CuprateMaterial, IronBasedMaterial
from .disorder import DisorderPattern, DisorderType
from .constants import PHYSICAL_CONSTANTS, MATERIAL_CONSTANTS

__all__ = [
    'Material',
    'CuprateMaterial', 
    'IronBasedMaterial',
    'DisorderPattern',
    'DisorderType',
    'PHYSICAL_CONSTANTS',
    'MATERIAL_CONSTANTS'
]