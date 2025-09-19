'use client'

import { useState, useEffect } from 'react'
import { format } from 'date-fns'

interface Experiment {
  id: string
  material: string
  temperature_k: number
  disorder_strength: number
  n_qubits: number
  backend: string
  tc_predicted?: number
  confidence_score?: number
  cost_euros: number
  created_at: string
  notes?: string
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [selectedMaterial, setSelectedMaterial] = useState('')

  useEffect(() => {
    fetchExperiments()
  }, [selectedMaterial])

  const fetchExperiments = async () => {
    try {
      const url = selectedMaterial 
        ? `/api/experiments?material=${selectedMaterial}`
        : '/api/experiments'
      const response = await fetch(url)
      const data = await response.json()
      setExperiments(data)
    } catch (error) {
      console.error('Error fetching experiments:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredExperiments = experiments.filter(exp =>
    exp.material.toLowerCase().includes(filter.toLowerCase()) ||
    exp.backend.toLowerCase().includes(filter.toLowerCase()) ||
    (exp.notes && exp.notes.toLowerCase().includes(filter.toLowerCase()))
  )

  const materials = [...new Set(experiments.map(exp => exp.material))]

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-lg text-gray-600">Loading experiments...</div>
      </div>
    )
  }

  return (
    <div className="px-4 py-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">All Experiments</h1>
        
        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <input
            type="text"
            placeholder="Search experiments..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-superconductor-500 focus:border-transparent"
          />
          
          <select
            value={selectedMaterial}
            onChange={(e) => setSelectedMaterial(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-superconductor-500 focus:border-transparent"
          >
            <option value="">All Materials</option>
            {materials.map(material => (
              <option key={material} value={material}>{material}</option>
            ))}
          </select>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-superconductor-600">
              {filteredExperiments.length}
            </div>
            <div className="text-sm text-gray-600">Filtered Results</div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-green-600">
              {filteredExperiments.filter(exp => exp.tc_predicted && exp.tc_predicted > 140).length}
            </div>
            <div className="text-sm text-gray-600">High-Tc Discoveries</div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-purple-600">
              {Math.max(...filteredExperiments.map(exp => exp.tc_predicted || 0)).toFixed(1)}K
            </div>
            <div className="text-sm text-gray-600">Best Tc</div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-orange-600">
              €{filteredExperiments.reduce((sum, exp) => sum + exp.cost_euros, 0).toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Total Cost</div>
          </div>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Material
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Parameters
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Results
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Backend
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Cost
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredExperiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{exp.material}</div>
                    {exp.notes && (
                      <div className="text-xs text-gray-500 truncate max-w-32" title={exp.notes}>
                        {exp.notes}
                      </div>
                    )}
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <div>T: {exp.temperature_k}K</div>
                    <div>Disorder: {exp.disorder_strength}</div>
                    <div>Qubits: {exp.n_qubits}</div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className={`text-sm font-medium ${
                      exp.tc_predicted && exp.tc_predicted > 140 ? 'text-green-600' : 'text-gray-900'
                    }`}>
                      Tc: {exp.tc_predicted?.toFixed(1) || 'N/A'}K
                    </div>
                    {exp.confidence_score && (
                      <div className="text-xs text-gray-500">
                        Confidence: {(exp.confidence_score * 100).toFixed(0)}%
                      </div>
                    )}
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      exp.backend === 'pasqal_cloud' 
                        ? 'bg-blue-100 text-blue-800'
                        : exp.backend === 'local_gpu'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {exp.backend}
                    </span>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    €{exp.cost_euros.toFixed(3)}
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {format(new Date(exp.created_at), 'MMM d, yyyy')}
                    <div className="text-xs">
                      {format(new Date(exp.created_at), 'HH:mm')}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {filteredExperiments.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">No experiments found</div>
            <div className="text-gray-400 text-sm mt-2">
              Try adjusting your filters or run some experiments
            </div>
          </div>
        )}
      </div>
    </div>
  )
}