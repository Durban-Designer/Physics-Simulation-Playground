"""
Quantum module for digital quantum circuits in strange metal simulation.

This module implements quantum entanglement circuits that complement the analog
Rydberg simulation to achieve the full tri-hybrid approach for superconductor discovery.
"""

from .strange_metal_entanglement import (
    QuantumEntanglementCircuit,
    StrangeMetalQuantumSimulation,
    EntanglementAnalyzer
)

__all__ = [
    'QuantumEntanglementCircuit',
    'StrangeMetalQuantumSimulation', 
    'EntanglementAnalyzer'
]