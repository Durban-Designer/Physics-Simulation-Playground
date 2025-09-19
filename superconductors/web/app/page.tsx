import Link from 'next/link'
import { getRecentExperiments, getExperimentStats } from './lib/database'

export default async function HomePage() {
  const recentExperiments = await getRecentExperiments(5)
  const stats = await getExperimentStats()

  return (
    <div className="px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Superconductor Research Laboratory
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Personal research platform for discovering novel superconductors through 
          quantum simulation and engineered disorder patterns.
        </p>
      </div>

      {/* Stats Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-2xl font-bold text-superconductor-600">
            {stats.totalExperiments}
          </div>
          <div className="text-sm text-gray-600">Total Experiments</div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-2xl font-bold text-green-600">
            {stats.discoveries}
          </div>
          <div className="text-sm text-gray-600">Discoveries (Tc > 140K)</div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-2xl font-bold text-purple-600">
            {stats.bestTc?.toFixed(1) || 'N/A'}K
          </div>
          <div className="text-sm text-gray-600">Highest Tc Predicted</div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-2xl font-bold text-orange-600">
            €{stats.totalCost?.toFixed(2) || '0.00'}
          </div>
          <div className="text-sm text-gray-600">Total Cost</div>
        </div>
      </div>

      {/* Recent Experiments */}
      <div className="bg-white rounded-lg shadow mb-8">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Experiments</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Material
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Temperature (K)
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Disorder
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Tc Predicted (K)
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Backend
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  When
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {recentExperiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{exp.material}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {exp.temperature_k}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {exp.disorder_strength}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-medium ${
                      exp.tc_predicted > 140 ? 'text-green-600' : 'text-gray-900'
                    }`}>
                      {exp.tc_predicted?.toFixed(1) || 'N/A'}
                    </span>
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
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(exp.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link 
          href="/experiments"
          className="bg-superconductor-600 text-white rounded-lg p-6 hover:bg-superconductor-700 transition-colors"
        >
          <h3 className="text-lg font-semibold mb-2">View All Experiments</h3>
          <p className="text-superconductor-100">
            Browse and analyze all your simulation runs
          </p>
        </Link>
        
        <Link 
          href="/discoveries"
          className="bg-green-600 text-white rounded-lg p-6 hover:bg-green-700 transition-colors"
        >
          <h3 className="text-lg font-semibold mb-2">Promising Discoveries</h3>
          <p className="text-green-100">
            Materials with Tc > 140K breakthrough potential
          </p>
        </Link>
        
        <Link 
          href="/analysis"
          className="bg-purple-600 text-white rounded-lg p-6 hover:bg-purple-700 transition-colors"
        >
          <h3 className="text-lg font-semibold mb-2">Data Analysis</h3>
          <p className="text-purple-100">
            Visualize trends and patterns in your data
          </p>
        </Link>
      </div>
    </div>
  )
}