import { NextResponse } from 'next/server'
import { getExperimentStats, getMaterialSummary } from '../../lib/database'

export async function GET() {
  try {
    const [stats, materialSummary] = await Promise.all([
      getExperimentStats(),
      getMaterialSummary()
    ])

    return NextResponse.json({
      ...stats,
      materialSummary
    })
  } catch (error) {
    console.error('Error fetching stats:', error)
    return NextResponse.json(
      { error: 'Failed to fetch statistics' },
      { status: 500 }
    )
  }
}