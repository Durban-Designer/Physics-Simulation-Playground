-- V001__initial_schema.sql
-- Initial schema for superconductor simulation tracking
-- This migration creates the core tables for tracking quantum simulations
-- with strict backend validation to ensure only real hardware results

-- Create enum types for backend validation
CREATE TYPE backend_type AS ENUM (
    'pasqal_cloud_emu_mps',  -- Real Pasqal Cloud EMU-MPS backend
    'cuquantum_gpu',         -- Real NVIDIA cuQuantum on GPU
    'qutip_cpu'              -- Real QuTiP CPU simulation
);

-- Create enum for simulation status tracking
CREATE TYPE simulation_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled'
);

-- Create enum for validation status
CREATE TYPE validation_status AS ENUM (
    'pending',
    'validated',
    'invalid',
    'experimental_comparison'
);

-- Main simulation runs table
CREATE TABLE simulation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backend_type backend_type NOT NULL,
    
    -- Material parameters
    material_name VARCHAR(100) NOT NULL,
    material_type VARCHAR(50) NOT NULL,
    lattice_size_x INTEGER NOT NULL,
    lattice_size_y INTEGER NOT NULL,
    
    -- Simulation parameters
    temperature_k REAL NOT NULL CHECK (temperature_k > 0),
    disorder_strength REAL NOT NULL CHECK (disorder_strength >= 0 AND disorder_strength <= 1),
    evolution_time_ns REAL NOT NULL CHECK (evolution_time_ns > 0),
    n_shots INTEGER NOT NULL CHECK (n_shots > 0),
    n_qubits INTEGER NOT NULL CHECK (n_qubits > 0 AND n_qubits <= 100),
    
    -- Timing and performance
    simulation_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    simulation_start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    simulation_end_time TIMESTAMP WITH TIME ZONE,
    simulation_duration_seconds REAL,
    
    -- Cost tracking
    cost_euros REAL CHECK (cost_euros >= 0),
    cost_per_shot REAL CHECK (cost_per_shot >= 0),
    
    -- Status tracking
    status simulation_status NOT NULL DEFAULT 'pending',
    validation_status validation_status NOT NULL DEFAULT 'pending',
    
    -- Result storage
    result_summary JSONB,  -- Key metrics: tc_predicted, order_parameter, etc.
    s3_result_path VARCHAR(500),  -- Full simulation data in S3
    result_size_mb REAL,
    
    -- Hardware details
    gpu_model VARCHAR(100),
    gpu_memory_gb REAL,
    cloud_region VARCHAR(50),
    cloud_job_id VARCHAR(200),
    
    -- Error tracking
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    workflow_id UUID,
    parent_simulation_id UUID REFERENCES simulation_runs(id),
    
    CONSTRAINT valid_backend CHECK (backend_type::text NOT LIKE '%mock%' AND backend_type::text NOT LIKE '%test%')
);

-- Index for backend type (excludes any mock/test backends)
CREATE INDEX idx_backend_real ON simulation_runs(backend_type) 
WHERE backend_type IN ('pasqal_cloud_emu_mps', 'cuquantum_gpu', 'qutip_cpu');

-- Performance indices
CREATE INDEX idx_simulation_timestamp ON simulation_runs(simulation_timestamp DESC);
CREATE INDEX idx_material_name ON simulation_runs(material_name);
CREATE INDEX idx_temperature ON simulation_runs(temperature_k);
CREATE INDEX idx_status ON simulation_runs(status) WHERE status != 'completed';
CREATE INDEX idx_workflow ON simulation_runs(workflow_id);

-- Material discoveries table
CREATE TABLE material_discoveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_run_id UUID NOT NULL REFERENCES simulation_runs(id),
    
    -- Discovery metrics
    tc_predicted_k REAL NOT NULL CHECK (tc_predicted_k > 0),
    tc_enhancement_factor REAL NOT NULL CHECK (tc_enhancement_factor > 0),
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    
    -- Material composition
    material_formula VARCHAR(200) NOT NULL,
    disorder_pattern JSONB NOT NULL,
    
    -- Validation
    experimental_validation BOOLEAN DEFAULT FALSE,
    experimental_tc_k REAL,
    experimental_reference VARCHAR(500),
    
    discovered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT tc_improvement CHECK (tc_predicted_k > 140)  -- Must beat current record
);

CREATE INDEX idx_tc_predicted ON material_discoveries(tc_predicted_k DESC);
CREATE INDEX idx_confidence ON material_discoveries(confidence_score DESC);

-- Cost tracking table
CREATE TABLE cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Time period
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Cost breakdown by backend
    pasqal_cloud_cost_euros REAL DEFAULT 0,
    gpu_compute_cost_euros REAL DEFAULT 0,
    storage_cost_euros REAL DEFAULT 0,
    total_cost_euros REAL GENERATED ALWAYS AS (
        COALESCE(pasqal_cloud_cost_euros, 0) + 
        COALESCE(gpu_compute_cost_euros, 0) + 
        COALESCE(storage_cost_euros, 0)
    ) STORED,
    
    -- Usage metrics
    pasqal_cloud_shots INTEGER DEFAULT 0,
    gpu_compute_hours REAL DEFAULT 0,
    storage_gb_hours REAL DEFAULT 0,
    
    -- Budget tracking
    budget_limit_euros REAL NOT NULL,
    budget_alert_threshold REAL DEFAULT 0.8,
    alert_sent BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cost_period ON cost_tracking(period_start, period_end);

-- Job queue table
CREATE TABLE job_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Job details
    job_type VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    
    -- Job configuration
    material_name VARCHAR(100) NOT NULL,
    simulation_config JSONB NOT NULL,
    backend_preference backend_type,
    
    -- Status tracking
    status simulation_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Resource requirements
    estimated_qubits INTEGER NOT NULL,
    estimated_duration_seconds REAL,
    estimated_cost_euros REAL,
    
    -- Execution tracking
    assigned_backend backend_type,
    simulation_run_id UUID REFERENCES simulation_runs(id),
    
    -- Error handling
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT
);

CREATE INDEX idx_queue_status_priority ON job_queue(status, priority DESC) WHERE status = 'pending';
CREATE INDEX idx_queue_created ON job_queue(created_at) WHERE status = 'pending';

-- Backend health table
CREATE TABLE backend_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backend_type backend_type NOT NULL,
    
    -- Health metrics
    is_available BOOLEAN NOT NULL,
    last_check_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    response_time_ms REAL,
    
    -- Performance metrics (rolling 24h)
    success_rate REAL CHECK (success_rate >= 0 AND success_rate <= 1),
    avg_simulation_time_seconds REAL,
    avg_cost_per_simulation REAL,
    
    -- Capacity
    current_load INTEGER DEFAULT 0,
    max_capacity INTEGER,
    
    -- Error tracking
    consecutive_failures INTEGER DEFAULT 0,
    last_error_message TEXT,
    last_error_time TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(backend_type)
);

-- Audit log for all operations
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- What happened
    operation_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(50) NOT NULL,
    record_id UUID,
    
    -- Who did it
    user_id VARCHAR(100) NOT NULL,
    ip_address INET,
    
    -- When it happened
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Details
    old_values JSONB,
    new_values JSONB,
    metadata JSONB
);

CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_table_record ON audit_log(table_name, record_id);

-- Create views for common queries

-- Active jobs view
CREATE VIEW active_jobs AS
SELECT 
    jq.*,
    sr.backend_type as actual_backend,
    sr.status as simulation_status,
    sr.cost_euros as actual_cost
FROM job_queue jq
LEFT JOIN simulation_runs sr ON jq.simulation_run_id = sr.id
WHERE jq.status IN ('pending', 'running');

-- Cost summary view
CREATE VIEW cost_summary AS
SELECT 
    DATE(simulation_timestamp) as date,
    backend_type,
    COUNT(*) as simulation_count,
    SUM(cost_euros) as total_cost,
    AVG(cost_euros) as avg_cost,
    SUM(n_shots) as total_shots
FROM simulation_runs
WHERE status = 'completed'
GROUP BY DATE(simulation_timestamp), backend_type;

-- Discovery candidates view
CREATE VIEW discovery_candidates AS
SELECT 
    md.*,
    sr.backend_type,
    sr.temperature_k,
    sr.disorder_strength,
    sr.simulation_timestamp
FROM material_discoveries md
JOIN simulation_runs sr ON md.simulation_run_id = sr.id
WHERE md.confidence_score > 0.8
    AND md.experimental_validation = FALSE
ORDER BY md.tc_predicted_k DESC;

-- Add comments for documentation
COMMENT ON TABLE simulation_runs IS 'Core table tracking all quantum simulation runs with strict backend validation';
COMMENT ON COLUMN simulation_runs.backend_type IS 'Real hardware backend used - no mocks allowed in production';
COMMENT ON COLUMN simulation_runs.cost_euros IS 'Actual cost from cloud provider API';
COMMENT ON COLUMN simulation_runs.validation_status IS 'Tracks if results have been validated against experiments';

COMMENT ON TABLE material_discoveries IS 'Promising superconductor candidates discovered through simulations';
COMMENT ON COLUMN material_discoveries.tc_predicted_k IS 'Predicted critical temperature - must exceed 140K';

COMMENT ON TABLE cost_tracking IS 'Budget monitoring and cost control for cloud resources';
COMMENT ON COLUMN cost_tracking.budget_alert_threshold IS 'Fraction of budget that triggers alerts (default 80%)';