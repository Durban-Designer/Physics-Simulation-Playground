-- V002__add_backend_validation_functions.sql
-- Functions and triggers for backend validation and data integrity

-- Function to validate backend is real (not mock/test)
CREATE OR REPLACE FUNCTION validate_backend_type()
RETURNS TRIGGER AS $$
BEGIN
    -- Reject any backend that contains mock, test, or simulated
    IF NEW.backend_type::text ~* '(mock|test|sim|fake|dummy)' THEN
        RAISE EXCEPTION 'Mock/test backends not allowed in production. Backend: %', NEW.backend_type;
    END IF;
    
    -- Ensure backend is one of the approved real backends
    IF NEW.backend_type NOT IN ('pasqal_cloud_emu_mps', 'cuquantum_gpu', 'qutip_cpu') THEN
        RAISE EXCEPTION 'Invalid backend type: %. Must be one of: pasqal_cloud_emu_mps, cuquantum_gpu, qutip_cpu', NEW.backend_type;
    END IF;
    
    -- Validate backend-specific constraints
    CASE NEW.backend_type
        WHEN 'pasqal_cloud_emu_mps' THEN
            IF NEW.cloud_job_id IS NULL THEN
                RAISE EXCEPTION 'Pasqal Cloud backend requires cloud_job_id';
            END IF;
            IF NEW.cost_euros IS NULL OR NEW.cost_euros <= 0 THEN
                RAISE EXCEPTION 'Pasqal Cloud backend must have positive cost';
            END IF;
            
        WHEN 'cuquantum_gpu' THEN
            IF NEW.gpu_model IS NULL OR NEW.gpu_memory_gb IS NULL THEN
                RAISE EXCEPTION 'GPU backend requires gpu_model and gpu_memory_gb';
            END IF;
            
        WHEN 'qutip_cpu' THEN
            IF NEW.n_qubits > 20 THEN
                RAISE EXCEPTION 'CPU backend limited to 20 qubits, got %', NEW.n_qubits;
            END IF;
    END CASE;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for backend validation
CREATE TRIGGER validate_backend_before_insert
BEFORE INSERT OR UPDATE ON simulation_runs
FOR EACH ROW
EXECUTE FUNCTION validate_backend_type();

-- Function to calculate simulation duration
CREATE OR REPLACE FUNCTION calculate_simulation_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.simulation_end_time IS NOT NULL AND NEW.simulation_start_time IS NOT NULL THEN
        NEW.simulation_duration_seconds := EXTRACT(EPOCH FROM (NEW.simulation_end_time - NEW.simulation_start_time));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for duration calculation
CREATE TRIGGER calculate_duration_before_update
BEFORE UPDATE ON simulation_runs
FOR EACH ROW
WHEN (NEW.simulation_end_time IS DISTINCT FROM OLD.simulation_end_time)
EXECUTE FUNCTION calculate_simulation_duration();

-- Function to check budget limits
CREATE OR REPLACE FUNCTION check_budget_limit()
RETURNS TRIGGER AS $$
DECLARE
    current_spending REAL;
    budget_limit REAL;
    alert_threshold REAL;
BEGIN
    -- Get current period spending
    SELECT 
        COALESCE(SUM(cost_euros), 0),
        MAX(ct.budget_limit_euros),
        MAX(ct.budget_alert_threshold)
    INTO current_spending, budget_limit, alert_threshold
    FROM simulation_runs sr
    CROSS JOIN LATERAL (
        SELECT * FROM cost_tracking 
        WHERE NOW() BETWEEN period_start AND period_end
        ORDER BY created_at DESC 
        LIMIT 1
    ) ct
    WHERE sr.simulation_timestamp >= ct.period_start
        AND sr.status = 'completed';
    
    -- Check if we're over budget
    IF budget_limit IS NOT NULL AND current_spending + NEW.cost_euros > budget_limit THEN
        RAISE EXCEPTION 'Budget limit exceeded. Current: €%, New cost: €%, Limit: €%', 
            current_spending, NEW.cost_euros, budget_limit;
    END IF;
    
    -- Check if we need to send alert
    IF budget_limit IS NOT NULL AND alert_threshold IS NOT NULL THEN
        IF current_spending + NEW.cost_euros > budget_limit * alert_threshold THEN
            -- Set flag for alert (actual alert sent by separate process)
            UPDATE cost_tracking 
            SET alert_sent = TRUE
            WHERE NOW() BETWEEN period_start AND period_end;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for budget checking
CREATE TRIGGER check_budget_before_simulation
BEFORE INSERT ON simulation_runs
FOR EACH ROW
WHEN (NEW.cost_euros IS NOT NULL)
EXECUTE FUNCTION check_budget_limit();

-- Function to update backend health metrics
CREATE OR REPLACE FUNCTION update_backend_health()
RETURNS TRIGGER AS $$
BEGIN
    -- Update backend health based on completed simulation
    IF NEW.status = 'completed' AND OLD.status != 'completed' THEN
        INSERT INTO backend_health (
            backend_type,
            is_available,
            success_rate,
            avg_simulation_time_seconds,
            avg_cost_per_simulation,
            consecutive_failures
        )
        VALUES (
            NEW.backend_type,
            TRUE,
            1.0,
            NEW.simulation_duration_seconds,
            NEW.cost_euros,
            0
        )
        ON CONFLICT (backend_type) DO UPDATE
        SET 
            is_available = TRUE,
            last_check_time = NOW(),
            success_rate = (
                SELECT COUNT(*)::REAL / NULLIF(COUNT(*), 0)
                FROM simulation_runs
                WHERE backend_type = NEW.backend_type
                    AND status = 'completed'
                    AND simulation_timestamp > NOW() - INTERVAL '24 hours'
            ),
            avg_simulation_time_seconds = (
                SELECT AVG(simulation_duration_seconds)
                FROM simulation_runs
                WHERE backend_type = NEW.backend_type
                    AND status = 'completed'
                    AND simulation_timestamp > NOW() - INTERVAL '24 hours'
            ),
            avg_cost_per_simulation = (
                SELECT AVG(cost_euros)
                FROM simulation_runs
                WHERE backend_type = NEW.backend_type
                    AND status = 'completed'
                    AND cost_euros IS NOT NULL
                    AND simulation_timestamp > NOW() - INTERVAL '24 hours'
            ),
            consecutive_failures = 0;
            
    ELSIF NEW.status = 'failed' AND OLD.status != 'failed' THEN
        -- Update failure metrics
        UPDATE backend_health
        SET 
            consecutive_failures = consecutive_failures + 1,
            last_error_message = NEW.error_message,
            last_error_time = NOW(),
            is_available = CASE 
                WHEN consecutive_failures >= 3 THEN FALSE 
                ELSE is_available 
            END
        WHERE backend_type = NEW.backend_type;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for health updates
CREATE TRIGGER update_health_after_simulation
AFTER UPDATE ON simulation_runs
FOR EACH ROW
WHEN (NEW.status IS DISTINCT FROM OLD.status)
EXECUTE FUNCTION update_backend_health();

-- Function for audit logging
CREATE OR REPLACE FUNCTION audit_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (
        operation_type,
        table_name,
        record_id,
        user_id,
        old_values,
        new_values
    )
    VALUES (
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        COALESCE(current_setting('app.current_user', TRUE), 'system'),
        CASE WHEN TG_OP != 'INSERT' THEN row_to_json(OLD) END,
        CASE WHEN TG_OP != 'DELETE' THEN row_to_json(NEW) END
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for important tables
CREATE TRIGGER audit_simulation_runs
AFTER INSERT OR UPDATE OR DELETE ON simulation_runs
FOR EACH ROW
EXECUTE FUNCTION audit_changes();

CREATE TRIGGER audit_material_discoveries
AFTER INSERT OR UPDATE OR DELETE ON material_discoveries
FOR EACH ROW
EXECUTE FUNCTION audit_changes();

CREATE TRIGGER audit_cost_tracking
AFTER INSERT OR UPDATE OR DELETE ON cost_tracking
FOR EACH ROW
EXECUTE FUNCTION audit_changes();

-- Function to validate material discovery
CREATE OR REPLACE FUNCTION validate_material_discovery()
RETURNS TRIGGER AS $$
DECLARE
    sim_backend backend_type;
BEGIN
    -- Get the backend type from the simulation
    SELECT backend_type INTO sim_backend
    FROM simulation_runs
    WHERE id = NEW.simulation_run_id;
    
    -- Ensure discovery comes from real backend
    IF sim_backend IS NULL THEN
        RAISE EXCEPTION 'Simulation run % not found', NEW.simulation_run_id;
    END IF;
    
    -- Validate Tc improvement
    IF NEW.tc_predicted_k <= 140 THEN
        RAISE EXCEPTION 'Discovery must exceed current record of 140K. Got: %K', NEW.tc_predicted_k;
    END IF;
    
    -- Validate confidence score
    IF NEW.confidence_score < 0.5 THEN
        RAISE WARNING 'Low confidence score: %. Consider additional validation.', NEW.confidence_score;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for discovery validation
CREATE TRIGGER validate_discovery_before_insert
BEFORE INSERT ON material_discoveries
FOR EACH ROW
EXECUTE FUNCTION validate_material_discovery();

-- Function to manage job queue priorities
CREATE OR REPLACE FUNCTION adjust_job_priority()
RETURNS void AS $$
BEGIN
    -- Increase priority for jobs that have been waiting too long
    UPDATE job_queue
    SET priority = LEAST(priority + 1, 10)
    WHERE status = 'pending'
        AND created_at < NOW() - INTERVAL '1 hour'
        AND priority < 10;
    
    -- Decrease priority for repeatedly failing jobs
    UPDATE job_queue
    SET priority = GREATEST(priority - 1, 1)
    WHERE status = 'pending'
        AND retry_count > 2
        AND priority > 1;
END;
$$ LANGUAGE plpgsql;

-- Add helpful utility functions

-- Get current spending for active period
CREATE OR REPLACE FUNCTION get_current_spending()
RETURNS TABLE (
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    total_spent REAL,
    budget_limit REAL,
    percentage_used REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH current_period AS (
        SELECT * FROM cost_tracking
        WHERE NOW() BETWEEN period_start AND period_end
        ORDER BY created_at DESC
        LIMIT 1
    )
    SELECT 
        cp.period_start,
        cp.period_end,
        COALESCE(SUM(sr.cost_euros), 0) as total_spent,
        cp.budget_limit_euros,
        CASE 
            WHEN cp.budget_limit_euros > 0 THEN 
                (COALESCE(SUM(sr.cost_euros), 0) / cp.budget_limit_euros * 100)
            ELSE 0
        END as percentage_used
    FROM current_period cp
    LEFT JOIN simulation_runs sr ON sr.simulation_timestamp BETWEEN cp.period_start AND cp.period_end
        AND sr.status = 'completed'
    GROUP BY cp.period_start, cp.period_end, cp.budget_limit_euros;
END;
$$ LANGUAGE plpgsql;

-- Get backend availability
CREATE OR REPLACE FUNCTION get_available_backends()
RETURNS TABLE (
    backend backend_type,
    available BOOLEAN,
    current_load INTEGER,
    success_rate REAL,
    avg_cost REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        bh.backend_type,
        bh.is_available AND bh.consecutive_failures < 3,
        bh.current_load,
        bh.success_rate,
        bh.avg_cost_per_simulation
    FROM backend_health bh
    WHERE bh.last_check_time > NOW() - INTERVAL '5 minutes'
    ORDER BY bh.success_rate DESC NULLS LAST;
END;
$$ LANGUAGE plpgsql;