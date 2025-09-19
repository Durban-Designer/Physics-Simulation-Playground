import { NextRequest, NextResponse } from 'next/server'
import { getRecentExperiments, insertExperiment, getExperimentsByMaterial } from '../../lib/database'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const material = searchParams.get('material')
  const limit = parseInt(searchParams.get('limit') || '50')

  try {
    let experiments
    if (material) {
      experiments = await getExperimentsByMaterial(material)
    } else {
      experiments = await getRecentExperiments(limit)
    }

    return NextResponse.json(experiments)
  } catch (error) {
    console.error('Error fetching experiments:', error)
    return NextResponse.json(
      { error: 'Failed to fetch experiments' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Validate required fields
    const requiredFields = ['material', 'temperature_k', 'disorder_strength', 'n_qubits', 'backend', 'result']
    for (const field of requiredFields) {
      if (!(field in body)) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        )
      }
    }

    // Insert experiment
    const experimentId = await insertExperiment(body)
    
    if (!experimentId) {
      return NextResponse.json(
        { error: 'Failed to create experiment' },
        { status: 500 }
      )
    }

    return NextResponse.json(
      { id: experimentId, message: 'Experiment created successfully' },
      { status: 201 }
    )
  } catch (error) {
    console.error('Error creating experiment:', error)
    return NextResponse.json(
      { error: 'Failed to create experiment' },
      { status: 500 }
    )
  }
}