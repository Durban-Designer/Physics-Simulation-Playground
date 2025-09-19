import { Pool } from 'pg'

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:research123@localhost:5432/experiments',
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
})

export interface Experiment {
  id: string
  material: string
  material_formula?: string
  temperature_k: number
  disorder_strength: number
  n_qubits: number
  backend: string
  simulation_time_seconds?: number
  result: any
  cost_euros: number
  shots: number
  created_at: string
  notes?: string
  tc_predicted?: number
  confidence_score?: number
}

export interface ExperimentStats {
  totalExperiments: number
  discoveries: number
  bestTc?: number
  totalCost?: number
  byBackend: { backend: string; count: number }[]
}

export async function getRecentExperiments(limit = 50): Promise<Experiment[]> {
  try {
    const result = await pool.query(`
      SELECT 
        id,
        material,
        material_formula,
        temperature_k,
        disorder_strength,
        n_qubits,
        backend,
        simulation_time_seconds,
        result,
        cost_euros,
        shots,
        created_at,
        notes,
        (result->>'tc_predicted')::REAL as tc_predicted,
        (result->>'confidence_score')::REAL as confidence_score
      FROM experiments
      ORDER BY created_at DESC
      LIMIT $1
    `, [limit])
    
    return result.rows
  } catch (error) {
    console.error('Error fetching recent experiments:', error)
    return []
  }
}

export async function getExperimentStats(): Promise<ExperimentStats> {
  try {
    const [totalResult, discoveryResult, bestTcResult, costResult, backendResult] = await Promise.all([
      pool.query('SELECT COUNT(*) as total FROM experiments'),
      pool.query("SELECT COUNT(*) as discoveries FROM experiments WHERE (result->>'tc_predicted')::REAL > 140"),
      pool.query("SELECT MAX((result->>'tc_predicted')::REAL) as best_tc FROM experiments WHERE result->>'tc_predicted' IS NOT NULL"),
      pool.query('SELECT SUM(cost_euros) as total_cost FROM experiments'),
      pool.query('SELECT backend, COUNT(*) as count FROM experiments GROUP BY backend ORDER BY count DESC')
    ])

    return {
      totalExperiments: parseInt(totalResult.rows[0].total),
      discoveries: parseInt(discoveryResult.rows[0].discoveries),
      bestTc: bestTcResult.rows[0].best_tc ? parseFloat(bestTcResult.rows[0].best_tc) : undefined,
      totalCost: costResult.rows[0].total_cost ? parseFloat(costResult.rows[0].total_cost) : undefined,
      byBackend: backendResult.rows
    }
  } catch (error) {
    console.error('Error fetching experiment stats:', error)
    return {
      totalExperiments: 0,
      discoveries: 0,
      byBackend: []
    }
  }
}

export async function getExperimentsByMaterial(material: string): Promise<Experiment[]> {
  try {
    const result = await pool.query(`
      SELECT 
        id,
        material,
        material_formula,
        temperature_k,
        disorder_strength,
        n_qubits,
        backend,
        simulation_time_seconds,
        result,
        cost_euros,
        shots,
        created_at,
        notes,
        (result->>'tc_predicted')::REAL as tc_predicted,
        (result->>'confidence_score')::REAL as confidence_score
      FROM experiments
      WHERE material = $1
      ORDER BY created_at DESC
    `, [material])
    
    return result.rows
  } catch (error) {
    console.error('Error fetching experiments by material:', error)
    return []
  }
}

export async function getPromaisingDiscoveries(): Promise<Experiment[]> {
  try {
    const result = await pool.query(`
      SELECT 
        id,
        material,
        material_formula,
        temperature_k,
        disorder_strength,
        n_qubits,
        backend,
        simulation_time_seconds,
        result,
        cost_euros,
        shots,
        created_at,
        notes,
        (result->>'tc_predicted')::REAL as tc_predicted,
        (result->>'confidence_score')::REAL as confidence_score
      FROM experiments
      WHERE (result->>'tc_predicted')::REAL > 140
      ORDER BY (result->>'tc_predicted')::REAL DESC
    `)
    
    return result.rows
  } catch (error) {
    console.error('Error fetching promising discoveries:', error)
    return []
  }
}

export async function getMaterialSummary() {
  try {
    const result = await pool.query(`
      SELECT 
        material,
        COUNT(*) as experiment_count,
        AVG((result->>'tc_predicted')::REAL) as avg_tc,
        MAX((result->>'tc_predicted')::REAL) as best_tc,
        SUM(cost_euros) as total_cost,
        AVG(disorder_strength) as avg_disorder,
        AVG(temperature_k) as avg_temperature
      FROM experiments
      WHERE result->>'tc_predicted' IS NOT NULL
      GROUP BY material
      ORDER BY best_tc DESC NULLS LAST
    `)
    
    return result.rows
  } catch (error) {
    console.error('Error fetching material summary:', error)
    return []
  }
}

export async function insertExperiment(experiment: Omit<Experiment, 'id' | 'created_at'>): Promise<string | null> {
  try {
    const result = await pool.query(`
      INSERT INTO experiments (
        material, material_formula, temperature_k, disorder_strength,
        n_qubits, backend, simulation_time_seconds, result,
        cost_euros, shots, notes
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      RETURNING id
    `, [
      experiment.material,
      experiment.material_formula,
      experiment.temperature_k,
      experiment.disorder_strength,
      experiment.n_qubits,
      experiment.backend,
      experiment.simulation_time_seconds,
      JSON.stringify(experiment.result),
      experiment.cost_euros,
      experiment.shots,
      experiment.notes
    ])
    
    return result.rows[0].id
  } catch (error) {
    console.error('Error inserting experiment:', error)
    return null
  }
}

// Helper function to analyze temperature-disorder space for a material
export async function getTemperatureDisorderAnalysis(material: string) {
  try {
    const result = await pool.query(`
      SELECT 
        temperature_k,
        disorder_strength,
        AVG((result->>'tc_predicted')::REAL) as avg_tc,
        COUNT(*) as run_count,
        MAX((result->>'tc_predicted')::REAL) as best_tc,
        AVG((result->>'confidence_score')::REAL) as avg_confidence
      FROM experiments
      WHERE material = $1 
        AND result->>'tc_predicted' IS NOT NULL
      GROUP BY temperature_k, disorder_strength
      ORDER BY avg_tc DESC
    `, [material])
    
    return result.rows
  } catch (error) {
    console.error('Error fetching temperature-disorder analysis:', error)
    return []
  }
}