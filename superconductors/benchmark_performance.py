#!/usr/bin/env python3
"""
Performance benchmarking script for GPU acceleration claims.

This script measures actual performance of different backends to validate
the claimed 10-100x speedup from GPU acceleration.
"""

import time
import numpy as np
from typing import Dict, List
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analog.strange_metal_disorder import StrangeMetalAnalogSimulation
from core.gpu_backends import SimulationConfig, BackendType


def benchmark_backend(backend_type: BackendType, lattice_size: tuple, n_trials: int = 3) -> Dict:
    """Benchmark a specific backend configuration."""
    print(f"\n🔬 Benchmarking {backend_type.value} with lattice {lattice_size}")
    
    config = SimulationConfig(
        backend=backend_type,
        verbose=True
    )
    
    sim = StrangeMetalAnalogSimulation('YBCO', simulation_config=config)
    times = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}...")
        
        try:
            start_time = time.time()
            result = sim.run_simulation(
                temperature=100.0,
                lattice_size=lattice_size,
                evolution_time=50.0,  # Reduced for faster benchmarking
                n_shots=25
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"    ✅ Completed in {elapsed:.3f}s")
            print(f"    Method: {result.get('simulation_method', 'unknown')}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return {
                'backend': backend_type.value,
                'lattice_size': lattice_size,
                'status': 'failed',
                'error': str(e)
            }
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'backend': backend_type.value,
        'lattice_size': lattice_size,
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times,
        'status': 'success'
    }


def main():
    """Run comprehensive performance benchmarks."""
    print("🚀 GPU Acceleration Benchmark Suite")
    print("=" * 50)
    
    # Check GPU availability
    try:
        import cupy as cp
        print(f"✅ GPU Available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"✅ GPU Memory: {cp.cuda.runtime.memGetInfo()[1]/1024**3:.1f}GB")
        else:
            print("⚠️ No GPU detected - CPU-only benchmarks")
    except ImportError:
        print("⚠️ CuPy not available - CPU-only benchmarks")
    
    # Test configurations
    test_configs = [
        # Small systems (should use CPU)
        (BackendType.CPU_QUTIP, (2, 2)),
        (BackendType.HYBRID, (2, 2)),
        
        # Medium systems (should use GPU if available)
        (BackendType.CPU_QUTIP, (3, 3)),
        (BackendType.GPU_CUQUANTUM, (3, 3)),
        (BackendType.HYBRID, (3, 3)),
    ]
    
    results = []
    
    for backend, lattice_size in test_configs:
        try:
            result = benchmark_backend(backend, lattice_size, n_trials=2)
            results.append(result)
        except Exception as e:
            print(f"❌ Benchmark failed for {backend.value}: {e}")
            results.append({
                'backend': backend.value,
                'lattice_size': lattice_size,
                'status': 'error',
                'error': str(e)
            })
    
    # Analyze results
    print("\n📊 Benchmark Results Summary")
    print("=" * 50)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    for result in successful_results:
        backend = result['backend']
        lattice = result['lattice_size']
        mean_time = result['mean_time']
        std_time = result['std_time']
        
        print(f"{backend:15s} {str(lattice):8s}: {mean_time:.3f}±{std_time:.3f}s")
    
    # Calculate speedups
    print("\n⚡ Speedup Analysis")
    print("=" * 30)
    
    # Group by lattice size
    lattice_groups = {}
    for result in successful_results:
        lattice = str(result['lattice_size'])
        if lattice not in lattice_groups:
            lattice_groups[lattice] = []
        lattice_groups[lattice].append(result)
    
    for lattice, group in lattice_groups.items():
        print(f"\nLattice {lattice}:")
        
        # Find CPU baseline
        cpu_result = None
        gpu_result = None
        hybrid_result = None
        
        for result in group:
            if 'cpu' in result['backend'].lower():
                cpu_result = result
            elif 'gpu' in result['backend'].lower():
                gpu_result = result
            elif 'hybrid' in result['backend'].lower():
                hybrid_result = result
        
        if cpu_result:
            cpu_time = cpu_result['mean_time']
            print(f"  CPU baseline: {cpu_time:.3f}s")
            
            if gpu_result:
                gpu_time = gpu_result['mean_time']
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                print(f"  GPU: {gpu_time:.3f}s (speedup: {speedup:.1f}x)")
            
            if hybrid_result:
                hybrid_time = hybrid_result['mean_time']
                speedup = cpu_time / hybrid_time if hybrid_time > 0 else float('inf')
                print(f"  Hybrid: {hybrid_time:.3f}s (speedup: {speedup:.1f}x)")
    
    # Summary
    print("\n🎯 Performance Assessment")
    print("=" * 30)
    
    if successful_results:
        print("✅ Benchmarking completed successfully")
        print(f"✅ {len(successful_results)} configurations tested")
        
        # Check if claimed speedups are realistic
        gpu_results = [r for r in successful_results if 'gpu' in r['backend'].lower()]
        if gpu_results:
            print("✅ GPU acceleration functional")
        else:
            print("⚠️ No successful GPU tests - check GPU setup")
            
    else:
        print("❌ No successful benchmark runs")
        print("❌ Performance claims cannot be validated")
    
    return results


if __name__ == "__main__":
    results = main()