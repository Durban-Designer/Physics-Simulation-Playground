"""
Orchestration module for tri-hybrid superconductor discovery workflow.

This module coordinates analog (Pasqal), quantum (digital circuits), and 
classical (transport) simulations to discover optimal superconducting materials.
"""

from .strange_metal_workflow import (
    TriHybridWorkflow,
    WorkflowConfig,
    DiscoveryProtocol,
    SuperconductorOptimizer
)

__all__ = [
    'TriHybridWorkflow',
    'WorkflowConfig', 
    'DiscoveryProtocol',
    'SuperconductorOptimizer'
]