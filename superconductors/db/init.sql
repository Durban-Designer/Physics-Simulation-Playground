-- Initialize the experiments database
-- Simple schema for personal superconductor research

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main experiments table
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Material parameters
    material VARCHAR(50) NOT NULL,
    material_formula VARCHAR(100),
    
    -- Simulation parameters
    temperature_k REAL NOT NULL CHECK (temperature_k > 0),
    disorder_strength REAL NOT NULL CHECK (disorder_strength >= 0 AND disorder_strength <= 1),
    n_qubits INTEGER NOT NULL CHECK (n_qubits > 0 AND n_qubits <= 100),
    
    -- Backend information
    backend VARCHAR(50) NOT NULL CHECK (backend IN ('local_gpu', 'pasqal_cloud', 'cpu_fallback')),
    simulation_time_seconds REAL,
    
    -- Results (JSON for flexibility)
    result JSONB NOT NULL,
    
    -- Cost tracking
    cost_euros REAL DEFAULT 0 CHECK (cost_euros >= 0),
    shots INTEGER DEFAULT 1000,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- Useful indexes for research queries
CREATE INDEX idx_experiments_material ON experiments(material);
CREATE INDEX idx_experiments_created ON experiments(created_at DESC);
CREATE INDEX idx_experiments_backend ON experiments(backend);
CREATE INDEX idx_experiments_temperature ON experiments(temperature_k);
CREATE INDEX idx_experiments_disorder ON experiments(disorder_strength);

-- GIN index for JSON queries
CREATE INDEX idx_experiments_result ON experiments USING GIN (result);

-- View for quick experiment analysis
CREATE VIEW experiment_summary AS
SELECT 
    material,
    backend,
    COUNT(*) as total_runs,
    AVG(temperature_k) as avg_temperature,
    AVG(disorder_strength) as avg_disorder,
    AVG((result->>'tc_predicted')::REAL) as avg_tc_predicted,
    MIN((result->>'tc_predicted')::REAL) as min_tc,
    MAX((result->>'tc_predicted')::REAL) as max_tc,
    SUM(cost_euros) as total_cost,
    AVG(simulation_time_seconds) as avg_simulation_time
FROM experiments
WHERE result->>'tc_predicted' IS NOT NULL
GROUP BY material, backend
ORDER BY avg_tc_predicted DESC;

-- View for promising discoveries (Tc > 140K)
CREATE VIEW promising_discoveries AS
SELECT 
    id,
    material,
    temperature_k,
    disorder_strength,
    (result->>'tc_predicted')::REAL as tc_predicted,
    (result->>'confidence_score')::REAL as confidence,
    backend,
    created_at,
    notes
FROM experiments
WHERE (result->>'tc_predicted')::REAL > 140
ORDER BY (result->>'tc_predicted')::REAL DESC;

-- View for cost analysis
CREATE VIEW cost_analysis AS
SELECT 
    DATE(created_at) as experiment_date,
    backend,
    COUNT(*) as runs,
    SUM(cost_euros) as daily_cost,
    AVG(cost_euros) as avg_cost_per_run,
    SUM(shots) as total_shots
FROM experiments
GROUP BY DATE(created_at), backend
ORDER BY experiment_date DESC;

-- Function to get latest experiments
CREATE OR REPLACE FUNCTION get_recent_experiments(limit_count INTEGER DEFAULT 50)
RETURNS TABLE (
    id UUID,
    material VARCHAR,
    temperature_k REAL,
    disorder_strength REAL,
    tc_predicted REAL,
    confidence REAL,
    backend VARCHAR,
    cost_euros REAL,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.material,
        e.temperature_k,
        e.disorder_strength,
        (e.result->>'tc_predicted')::REAL,
        (e.result->>'confidence_score')::REAL,
        e.backend,
        e.cost_euros,
        e.created_at
    FROM experiments e
    ORDER BY e.created_at DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze material performance
CREATE OR REPLACE FUNCTION analyze_material(material_name VARCHAR)
RETURNS TABLE (
    temperature_k REAL,
    disorder_strength REAL,
    avg_tc REAL,
    run_count INTEGER,
    best_tc REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.temperature_k,
        e.disorder_strength,
        AVG((e.result->>'tc_predicted')::REAL) as avg_tc,
        COUNT(*)::INTEGER as run_count,
        MAX((e.result->>'tc_predicted')::REAL) as best_tc
    FROM experiments e
    WHERE e.material = material_name
        AND e.result->>'tc_predicted' IS NOT NULL
    GROUP BY e.temperature_k, e.disorder_strength
    ORDER BY avg_tc DESC;
END;
$$ LANGUAGE plpgsql;

-- Insert some example data for testing
INSERT INTO experiments (
    material, material_formula, temperature_k, disorder_strength, 
    n_qubits, backend, result, cost_euros, notes
) VALUES 
(
    'YBCO', 'YBa2Cu3O7',
    100.0, 0.05,
    16, 'local_gpu',
    '{"tc_predicted": 95.5, "confidence_score": 0.85, "order_parameter": 0.7}',
    0.0, 'Initial test run'
),
(
    'BSCCO', 'Bi2Sr2CaCu2O8',
    150.0, 0.1,
    20, 'local_gpu',
    '{"tc_predicted": 88.2, "confidence_score": 0.78, "order_parameter": 0.65}',
    0.0, 'Disorder effect study'
),
(
    'YBCO', 'YBa2Cu3O7',
    200.0, 0.08,
    25, 'local_simulation',
    NULL,
    0.0, 'Failed simulation - physics calculation required'
);

-- Add helpful comments
COMMENT ON TABLE experiments IS 'Main table storing all superconductor simulation experiments';
COMMENT ON COLUMN experiments.result IS 'JSON containing simulation results: tc_predicted, confidence_score, order_parameter, etc.';
COMMENT ON COLUMN experiments.backend IS 'Which computational backend was used: local_gpu, pasqal_cloud, cpu_fallback';
COMMENT ON VIEW promising_discoveries IS 'Experiments that discovered materials with Tc > 140K (current record)';

-- Grant permissions (for development)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;