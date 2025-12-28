import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

const MetricsPanel = ({ instanceState }) => {
  // Create stance data for bar chart
  const stanceData = Object.entries(instanceState.stance).map(([key, value]) => ({
    name: key,
    value: value * 100
  }))

  // Simulated history data for demo (in real app, this would come from API)
  const stabilityHistory = Array.from({ length: 20 }, (_, i) => ({
    interaction: i + 1,
    stability: Math.min(0.95, (i / 20) * 0.8 + Math.random() * 0.1),
    plasticity: Math.max(0.05, 1 - (i / 20) * 0.8 - Math.random() * 0.1)
  }))

  return (
    <div className="p-6 space-y-8 overflow-y-auto h-full">
      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Total Interactions"
          value={instanceState.interaction_count}
          color="text-blue-400"
        />
        <StatCard
          label="Significant"
          value={instanceState.significant_count}
          color="text-amber-400"
        />
        <StatCard
          label="Eval Rate"
          value={instanceState.interaction_count > 0 
            ? `${((instanceState.significant_count / instanceState.interaction_count) * 100).toFixed(1)}%`
            : '0%'
          }
          color="text-purple-400"
        />
        <StatCard
          label="Current Threshold"
          value={`${(instanceState.current_threshold * 100).toFixed(1)}%`}
          color="text-emerald-400"
        />
      </div>

      {/* Stance Bar Chart */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Stance Dimensions</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stanceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8' }} />
              <YAxis 
                type="category" 
                dataKey="name" 
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                width={100}
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #475569',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Bar 
                dataKey="value" 
                fill="#a855f7" 
                radius={[0, 4, 4, 0]}
                background={{ fill: '#1e293b' }}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Stability/Plasticity Over Time (Demo) */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          Stability & Plasticity Dynamics
          <span className="text-xs text-slate-500 ml-2">(simulated trajectory)</span>
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={stabilityHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="interaction" 
                tick={{ fill: '#94a3b8' }}
                label={{ value: 'Interactions', position: 'bottom', fill: '#64748b' }}
              />
              <YAxis 
                domain={[0, 1]} 
                tick={{ fill: '#94a3b8' }}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #475569',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#e2e8f0' }}
                formatter={(value) => [`${(value * 100).toFixed(1)}%`]}
              />
              <Line 
                type="monotone" 
                dataKey="stability" 
                stroke="#10b981" 
                strokeWidth={2}
                dot={false}
                name="Stability"
              />
              <Line 
                type="monotone" 
                dataKey="plasticity" 
                stroke="#f59e0b" 
                strokeWidth={2}
                dot={false}
                name="Plasticity"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-center gap-6 mt-2">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-emerald-500 rounded"></span>
            <span className="text-sm text-slate-400">Stability</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 bg-amber-500 rounded"></span>
            <span className="text-sm text-slate-400">Plasticity</span>
          </div>
        </div>
      </div>

      {/* Phase Progression */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Phase Progression</h3>
        <div className="flex items-center">
          {['rapid_formation', 'consolidation', 'stabilization', 'maturity'].map((phase, i) => {
            const isActive = instanceState.phase === phase
            const isPast = ['rapid_formation', 'consolidation', 'stabilization', 'maturity']
              .indexOf(instanceState.phase) > i
            
            return (
              <React.Fragment key={phase}>
                <div className={`flex-1 text-center ${i > 0 ? 'border-l border-slate-600' : ''}`}>
                  <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                    isActive ? 'bg-purple-500 ring-2 ring-purple-400 ring-offset-2 ring-offset-slate-800' :
                    isPast ? 'bg-emerald-500' : 'bg-slate-600'
                  }`}></div>
                  <div className={`text-xs ${isActive ? 'text-purple-400 font-medium' : 'text-slate-500'}`}>
                    {phase.replace('_', ' ')}
                  </div>
                </div>
              </React.Fragment>
            )
          })}
        </div>
        <div className="mt-4 text-sm text-slate-400">
          <p><strong>Current:</strong> {instanceState.phase.replace('_', ' ')}</p>
          <p className="text-xs text-slate-500 mt-1">
            Stability threshold for maturity: 95% (sustained over 10 significant interactions)
          </p>
        </div>
      </div>

      {/* Environment Summary */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Environment Model</h3>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(instanceState.environment)
            .filter(([key]) => key !== 'key_traits')
            .map(([key, value]) => (
              <div key={key} className="bg-slate-700/50 rounded p-3">
                <div className="text-xs text-slate-500 mb-1">{key.replace(/_/g, ' ')}</div>
                <div className="text-slate-200 font-medium">{value}</div>
              </div>
            ))}
        </div>
        {instanceState.environment.key_traits?.length > 0 && (
          <div className="mt-4">
            <div className="text-xs text-slate-500 mb-2">Key Traits Observed</div>
            <div className="flex flex-wrap gap-2">
              {instanceState.environment.key_traits.map((trait, i) => (
                <span key={i} className="px-3 py-1 bg-slate-700 rounded-full text-sm text-slate-300">
                  {trait}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const StatCard = ({ label, value, color }) => (
  <div className="bg-slate-800 rounded-lg p-4">
    <div className="text-xs text-slate-500 mb-1">{label}</div>
    <div className={`text-2xl font-bold ${color}`}>{value}</div>
  </div>
)

export default MetricsPanel
