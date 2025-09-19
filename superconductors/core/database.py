"""
PostgreSQL database connection and management for production simulation tracking.

This module provides database connectivity with connection pooling, automatic retries,
and comprehensive error handling for the superconductor discovery platform.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager
from uuid import UUID
import json

import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, ISOLATION_LEVEL_READ_COMMITTED
import psycopg2.errors

# Enable UUID support
psycopg2.extras.register_uuid()

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""
    
    def __init__(self):
        self.host = os.environ.get('DB_HOST', 'localhost')
        self.port = int(os.environ.get('DB_PORT', '5432'))
        self.name = os.environ.get('DB_NAME', 'superconductor_discovery')
        self.user = os.environ.get('DB_USER', 'postgres')
        self.password = os.environ.get('DB_PASSWORD', '')
        self.min_conn = int(os.environ.get('DB_MIN_CONN', '2'))
        self.max_conn = int(os.environ.get('DB_MAX_CONN', '10'))
        
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class BackendValidator:
    """Validates that only real hardware backends are used in production."""
    
    VALID_BACKENDS = {
        'pasqal_cloud_emu_mps',
        'cuquantum_gpu', 
        'qutip_cpu'
    }
    
    INVALID_PATTERNS = [
        'mock', 'test', 'sim', 'fake', 'dummy', 'stub'
    ]
    
    @classmethod
    def validate_backend(cls, backend: str) -> bool:
        """
        Validate that backend is a real hardware backend.
        
        Args:
            backend: Backend type string
            
        Returns:
            True if valid, raises exception if invalid
            
        Raises:
            ValueError: If backend is not a valid real hardware backend
        """
        if backend not in cls.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be one of: {', '.join(cls.VALID_BACKENDS)}"
            )
        
        # Double-check for any mock/test patterns
        backend_lower = backend.lower()
        for pattern in cls.INVALID_PATTERNS:
            if pattern in backend_lower:
                raise ValueError(
                    f"Mock/test backends not allowed in production. "
                    f"Backend '{backend}' contains forbidden pattern '{pattern}'"
                )
        
        return True


class DatabaseManager:
    """Manages PostgreSQL connections and operations for the discovery platform."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager with connection pool."""
        self.config = config or DatabaseConfig()
        self._pool = None
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize connection pool with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=self.config.min_conn,
                    maxconn=self.config.max_conn,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.name,
                    user=self.config.user,
                    password=self.config.password,
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                logger.info(f"Database connection pool initialized successfully")
                return
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize connection pool (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                else:
                    raise
    
    @contextmanager
    def get_connection(self, autocommit: bool = False):
        """
        Get a database connection from the pool.
        
        Args:
            autocommit: Whether to enable autocommit mode
            
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = self._pool.getconn()
            if autocommit:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            else:
                conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
            
            yield conn
            
            if not autocommit:
                conn.commit()
                
        except psycopg2.Error as e:
            if conn and not autocommit:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def insert_simulation_run(self, 
                            backend_type: str,
                            material_name: str,
                            simulation_params: Dict[str, Any]) -> UUID:
        """
        Insert a new simulation run with backend validation.
        
        Args:
            backend_type: Type of backend (must be real hardware)
            material_name: Name of material being simulated
            simulation_params: Dictionary of simulation parameters
            
        Returns:
            UUID of created simulation run
            
        Raises:
            ValueError: If backend is invalid
            psycopg2.Error: If database operation fails
        """
        # Validate backend before inserting
        BackendValidator.validate_backend(backend_type)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO simulation_runs (
                        backend_type, material_name, material_type,
                        lattice_size_x, lattice_size_y,
                        temperature_k, disorder_strength,
                        evolution_time_ns, n_shots, n_qubits,
                        simulation_start_time, status,
                        gpu_model, gpu_memory_gb, cloud_job_id,
                        created_by
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    ) RETURNING id
                """)
                
                cur.execute(query, (
                    backend_type,
                    material_name,
                    simulation_params.get('material_type', 'cuprate'),
                    simulation_params.get('lattice_size_x', 2),
                    simulation_params.get('lattice_size_y', 2),
                    simulation_params['temperature_k'],
                    simulation_params['disorder_strength'],
                    simulation_params.get('evolution_time_ns', 100),
                    simulation_params.get('n_shots', 1000),
                    simulation_params['n_qubits'],
                    datetime.utcnow(),
                    'pending',
                    simulation_params.get('gpu_model'),
                    simulation_params.get('gpu_memory_gb'),
                    simulation_params.get('cloud_job_id'),
                    simulation_params.get('created_by', 'system')
                ))
                
                result = cur.fetchone()
                simulation_id = result['id']
                
                logger.info(f"Created simulation run {simulation_id} with backend {backend_type}")
                return simulation_id
    
    def update_simulation_result(self,
                               simulation_id: UUID,
                               status: str,
                               result_data: Optional[Dict[str, Any]] = None,
                               error_message: Optional[str] = None,
                               cost_euros: Optional[float] = None) -> bool:
        """
        Update simulation run with results.
        
        Args:
            simulation_id: UUID of simulation run
            status: New status (completed, failed, cancelled)
            result_data: Dictionary of result data
            error_message: Error message if failed
            cost_euros: Actual cost from cloud provider
            
        Returns:
            True if update successful
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = sql.SQL("""
                    UPDATE simulation_runs
                    SET status = %s,
                        simulation_end_time = %s,
                        result_summary = %s,
                        error_message = %s,
                        cost_euros = %s
                    WHERE id = %s
                """)
                
                cur.execute(query, (
                    status,
                    datetime.utcnow(),
                    json.dumps(result_data) if result_data else None,
                    error_message,
                    cost_euros,
                    simulation_id
                ))
                
                if cur.rowcount > 0:
                    logger.info(f"Updated simulation {simulation_id} with status {status}")
                    return True
                else:
                    logger.warning(f"No simulation found with id {simulation_id}")
                    return False
    
    def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending jobs from queue ordered by priority.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of pending job dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = sql.SQL("""
                    SELECT jq.*, bh.is_available as backend_available
                    FROM job_queue jq
                    LEFT JOIN backend_health bh ON jq.backend_preference = bh.backend_type
                    WHERE jq.status = 'pending'
                        AND jq.retry_count < jq.max_retries
                    ORDER BY jq.priority DESC, jq.created_at ASC
                    LIMIT %s
                """)
                
                cur.execute(query, (limit,))
                return cur.fetchall()
    
    def check_budget_status(self) -> Dict[str, Any]:
        """
        Check current budget status and spending.
        
        Returns:
            Dictionary with budget information
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM get_current_spending()")
                result = cur.fetchone()
                
                if result:
                    return {
                        'period_start': result['period_start'],
                        'period_end': result['period_end'],
                        'total_spent': float(result['total_spent']),
                        'budget_limit': float(result['budget_limit']),
                        'percentage_used': float(result['percentage_used']),
                        'budget_remaining': float(result['budget_limit']) - float(result['total_spent']),
                        'alert_threshold_reached': float(result['percentage_used']) >= 80
                    }
                else:
                    return {
                        'error': 'No active budget period found',
                        'total_spent': 0,
                        'budget_limit': 0
                    }
    
    def get_backend_health(self) -> List[Dict[str, Any]]:
        """
        Get health status of all backends.
        
        Returns:
            List of backend health dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM get_available_backends()")
                return cur.fetchall()
    
    def record_material_discovery(self,
                                simulation_id: UUID,
                                tc_predicted: float,
                                material_formula: str,
                                disorder_pattern: Dict[str, Any],
                                confidence_score: float) -> Optional[UUID]:
        """
        Record a promising material discovery.
        
        Args:
            simulation_id: UUID of simulation that found this material
            tc_predicted: Predicted critical temperature in Kelvin
            material_formula: Chemical formula of discovered material
            disorder_pattern: Dictionary describing disorder configuration
            confidence_score: Confidence in prediction (0-1)
            
        Returns:
            UUID of discovery record if Tc > 140K, None otherwise
        """
        if tc_predicted <= 140:
            logger.info(f"Material with Tc={tc_predicted}K does not exceed threshold")
            return None
        
        # Calculate enhancement factor based on base material
        tc_enhancement = tc_predicted / 93.0  # Relative to YBCO
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = sql.SQL("""
                    INSERT INTO material_discoveries (
                        simulation_run_id, tc_predicted_k,
                        tc_enhancement_factor, confidence_score,
                        material_formula, disorder_pattern
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """)
                
                cur.execute(query, (
                    simulation_id,
                    tc_predicted,
                    tc_enhancement,
                    confidence_score,
                    material_formula,
                    json.dumps(disorder_pattern)
                ))
                
                result = cur.fetchone()
                discovery_id = result['id']
                
                logger.info(
                    f"Recorded material discovery {discovery_id}: "
                    f"{material_formula} with Tc={tc_predicted}K"
                )
                return discovery_id
    
    def close(self):
        """Close all database connections."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")


# Singleton instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create singleton database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager