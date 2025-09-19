#!/usr/bin/env python3
"""
Simple test script to verify database connectivity and insert a test experiment.
This bypasses the complex dependencies and tests the core functionality.
"""

import json
import time
from datetime import datetime

# Try to import psycopg2, install if not available
try:
    import psycopg2
except ImportError:
    print("Installing psycopg2...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2

def test_database_connection():
    """Test basic database connectivity."""
    try:
        conn = psycopg2.connect(
            "postgresql://postgres:research123@localhost:5432/experiments"
        )
        print("✅ Database connection successful")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def insert_test_experiment(conn):
    """Insert a test experiment."""
    
    # Create test experiment data
    test_data = {
        'material': 'YBCO',
        'material_formula': 'YBa2Cu3O7',
        'temperature_k': 175.0,
        'disorder_strength': 0.06,
        'n_qubits': 16,
        'backend': 'local_gpu',
        'simulation_time_seconds': 0.125,
        'result': {
            'tc_predicted': 98.3,
            'confidence_score': 0.87,
            'order_parameter': 0.72,
            'disorder_enhancement_factor': 1.2,
            'simulation_metadata': {
                'backend_method': 'qutip_cpu',
                'shots': 1000
            }
        },
        'cost_euros': 0.0,
        'shots': 1000,
        'notes': 'Test experiment from Python script'
    }
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiments (
                    material, material_formula, temperature_k, disorder_strength,
                    n_qubits, backend, simulation_time_seconds, result,
                    cost_euros, shots, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
            """, (
                test_data['material'],
                test_data['material_formula'],
                test_data['temperature_k'],
                test_data['disorder_strength'],
                test_data['n_qubits'],
                test_data['backend'],
                test_data['simulation_time_seconds'],
                json.dumps(test_data['result']),
                test_data['cost_euros'],
                test_data['shots'],
                test_data['notes']
            ))
            
            result = cur.fetchone()
            experiment_id, created_at = result
            conn.commit()
            
            print(f"✅ Test experiment inserted successfully!")
            print(f"   ID: {experiment_id}")
            print(f"   Material: {test_data['material']}")
            print(f"   Temperature: {test_data['temperature_k']}K")
            print(f"   Predicted Tc: {test_data['result']['tc_predicted']}K")
            print(f"   Backend: {test_data['backend']}")
            print(f"   Created: {created_at}")
            
            return experiment_id
            
    except Exception as e:
        print(f"❌ Failed to insert experiment: {e}")
        conn.rollback()
        return None

def verify_experiment(conn, experiment_id):
    """Verify the experiment was stored correctly."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    material,
                    temperature_k,
                    disorder_strength,
                    (result->>'tc_predicted')::REAL as tc_predicted,
                    (result->>'confidence_score')::REAL as confidence,
                    backend,
                    cost_euros,
                    created_at,
                    notes
                FROM experiments 
                WHERE id = %s
            """, (experiment_id,))
            
            result = cur.fetchone()
            if result:
                print(f"\n✅ Experiment verification successful!")
                print(f"   Material: {result[0]}")
                print(f"   Temperature: {result[1]}K")
                print(f"   Disorder: {result[2]}")
                print(f"   Predicted Tc: {result[3]}K")
                print(f"   Confidence: {result[4]:.2f}")
                print(f"   Backend: {result[5]}")
                print(f"   Cost: €{result[6]}")
                print(f"   Notes: {result[8]}")
                return True
            else:
                print(f"❌ Experiment {experiment_id} not found")
                return False
                
    except Exception as e:
        print(f"❌ Failed to verify experiment: {e}")
        return False

def show_all_experiments(conn):
    """Show all experiments in the database."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    material,
                    temperature_k,
                    disorder_strength,
                    (result->>'tc_predicted')::REAL as tc_predicted,
                    backend,
                    created_at
                FROM experiments 
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            results = cur.fetchall()
            print(f"\n📊 All experiments in database ({len(results)} shown):")
            print("   Material | Temp(K) | Disorder | Tc(K) | Backend     | Created")
            print("   ---------|---------|----------|-------|-------------|--------")
            for row in results:
                material, temp, disorder, tc, backend, created = row
                created_str = created.strftime("%m/%d %H:%M")
                print(f"   {material:8} | {temp:7.1f} | {disorder:8.2f} | {tc:5.1f} | {backend:11} | {created_str}")
                
    except Exception as e:
        print(f"❌ Failed to show experiments: {e}")

def main():
    """Main test function."""
    print("🧪 Testing Superconductor Research Platform")
    print("=" * 50)
    
    # Test database connection
    conn = test_database_connection()
    if not conn:
        return
    
    try:
        # Insert test experiment
        experiment_id = insert_test_experiment(conn)
        if not experiment_id:
            return
        
        # Verify the experiment
        if verify_experiment(conn, experiment_id):
            print("\n🎉 Test completed successfully!")
        
        # Show all experiments
        show_all_experiments(conn)
        
        print(f"\n🌐 View results in web UI: http://localhost:3000")
        
    finally:
        conn.close()
        print("\n📝 Database connection closed")

if __name__ == "__main__":
    main()