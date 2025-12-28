import React, { useState, useEffect } from 'react'
import { Brain, Plus, Trash2, Send, Activity, Settings, MessageSquare, BarChart3, Key, Download, Beaker } from 'lucide-react'
import StanceRadar from './components/StanceRadar'
import PhaseIndicator from './components/PhaseIndicator'
import InteractionPanel from './components/InteractionPanel'
import MetricsPanel from './components/MetricsPanel'
import HistoryPanel from './components/HistoryPanel'
import ControlExperiment from './components/ControlExperiment'

const API_BASE = '/api'

function App() {
  const [instances, setInstances] = useState([])
  const [selectedInstance, setSelectedInstance] = useState(null)
  const [instanceState, setInstanceState] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('interact')
  const [sessionId] = useState(() => crypto.randomUUID())
  const [apiKeyConfigured, setApiKeyConfigured] = useState(false)

  useEffect(() => {
    fetchInstances()
    checkApiKey()
  }, [])

  const checkApiKey = async () => {
    try {
      const res = await fetch(`${API_BASE}/api-key/${sessionId}`)
      const data = await res.json()
      setApiKeyConfigured(data.configured)
    } catch (err) {
      console.error('Failed to check API key')
    }
  }

  useEffect(() => {
    // Only fetch if we don't already have the state for this instance
    if (selectedInstance && (!instanceState || instanceState.instance_id !== selectedInstance)) {
      fetchInstanceState(selectedInstance)
    }
  }, [selectedInstance])

  const fetchInstances = async () => {
    try {
      const res = await fetch(`${API_BASE}/instances`)
      const data = await res.json()
      setInstances(data)
      if (data.length > 0 && !selectedInstance) {
        setSelectedInstance(data[0])
      }
    } catch (err) {
      setError('Failed to fetch instances')
    }
  }

  const fetchInstanceState = async (instanceId) => {
    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/instances/${instanceId}`)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      // Only update if this is still the selected instance
      if (instanceId === selectedInstance) {
        setInstanceState(data)
      }
    } catch (err) {
      console.error('Failed to fetch instance state:', err)
      setError('Failed to fetch instance state')
    } finally {
      setLoading(false)
    }
  }

  const createInstance = async () => {
    try {
      const res = await fetch(`${API_BASE}/instances`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      })
      const data = await res.json()
      setInstances([...instances, data.instance_id])
      setSelectedInstance(data.instance_id)
      setInstanceState(data)
    } catch (err) {
      setError('Failed to create instance')
    }
  }

  const deleteInstance = async (instanceId) => {
    try {
      await fetch(`${API_BASE}/instances/${instanceId}`, { method: 'DELETE' })
      const newInstances = instances.filter(id => id !== instanceId)
      setInstances(newInstances)
      if (selectedInstance === instanceId) {
        setSelectedInstance(newInstances[0] || null)
        setInstanceState(null)
      }
    } catch (err) {
      setError('Failed to delete instance')
    }
  }

  const handleInteraction = async (response) => {
    // Only update if we got a valid state back
    if (response && response.state && response.state.instance_id) {
      setInstanceState(response.state)
    }
  }

  const exportMetrics = async () => {
    if (!selectedInstance) return
    
    try {
      const res = await fetch(`${API_BASE}/instances/${selectedInstance}/export`)
      const data = await res.json()
      
      // Create and download JSON file
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `nurture-metrics-${selectedInstance}-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError('Failed to export metrics')
    }
  }

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-purple-400" />
            <div>
              <h1 className="text-xl font-bold text-white">Nurture Layer</h1>
              <p className="text-sm text-slate-400">Runtime Character Formation System</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-slate-400">
              {instances.length} instance{instances.length !== 1 ? 's' : ''}
            </span>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-73px)]">
        {/* Sidebar */}
        <aside className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col">
          <div className="p-4 border-b border-slate-700">
            <button
              onClick={createInstance}
              className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Instance
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-2">
            {instances.map(id => (
              <div
                key={id}
                className={`flex items-center justify-between p-3 rounded-lg cursor-pointer mb-1 ${
                  selectedInstance === id 
                    ? 'bg-purple-600/20 border border-purple-500/50' 
                    : 'hover:bg-slate-700'
                }`}
                onClick={() => setSelectedInstance(id)}
              >
                <span className="text-sm font-mono text-slate-300 truncate">
                  {id.substring(0, 8)}...
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteInstance(id); }}
                  className="p-1 hover:bg-slate-600 rounded"
                >
                  <Trash2 className="w-4 h-4 text-slate-400 hover:text-red-400" />
                </button>
              </div>
            ))}
            {instances.length === 0 && (
              <p className="text-center text-slate-500 text-sm py-8">
                No instances yet.<br />Create one to begin.
              </p>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {selectedInstance && instanceState ? (
            <>
              {/* Instance Header */}
              <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white font-mono">
                      {selectedInstance}
                    </h2>
                    <div className="flex items-center gap-4 mt-1">
                      <PhaseIndicator phase={instanceState.phase} />
                      <span className="text-sm text-slate-400">
                        {instanceState.interaction_count} interactions
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-emerald-400">
                        {(instanceState.stability * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-slate-500">Stability</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-amber-400">
                        {(instanceState.plasticity * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-slate-500">Plasticity</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div className="bg-slate-800/30 border-b border-slate-700 px-6">
                <div className="flex justify-between items-center">
                  <div className="flex gap-1">
                    {[
                      { id: 'interact', icon: MessageSquare, label: 'Interact' },
                      { id: 'metrics', icon: BarChart3, label: 'Metrics' },
                      { id: 'history', icon: Activity, label: 'History' },
                      { id: 'control', icon: Beaker, label: 'Control Exp' },
                    ].map(tab => (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                          activeTab === tab.id
                            ? 'text-purple-400 border-b-2 border-purple-400'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={exportMetrics}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-emerald-400 hover:text-emerald-300 hover:bg-emerald-900/20 rounded transition-colors"
                    title="Export metrics as JSON for scientific analysis"
                  >
                    <Download className="w-4 h-4" />
                    Export JSON
                  </button>
                </div>
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-hidden">
                {activeTab === 'interact' && (
                  <div className="h-full flex">
                    <div className="flex-1 overflow-y-auto">
                      <InteractionPanel
                        instanceId={selectedInstance}
                        instanceState={instanceState}
                        onInteraction={handleInteraction}
                        apiBase={API_BASE}
                        sessionId={sessionId}
                        apiKeyConfigured={apiKeyConfigured}
                        onApiKeyChange={setApiKeyConfigured}
                      />
                    </div>
                    <div className="w-96 border-l border-slate-800/50 p-4 overflow-y-auto">
                      <h3 className="text-sm font-semibold text-slate-300 mb-4">Relational Stance</h3>
                      <StanceRadar stance={instanceState.stance} />
                      <div className="mt-6">
                        <h3 className="text-sm font-semibold text-slate-300 mb-3">Environment Model</h3>
                        <div className="space-y-2 text-sm">
                          {Object.entries(instanceState.environment).map(([key, value]) => (
                            key !== 'key_traits' && (
                              <div key={key} className="flex justify-between">
                                <span className="text-slate-500">{key.replace(/_/g, ' ')}</span>
                                <span className="text-slate-300">{value}</span>
                              </div>
                            )
                          ))}
                        </div>
                        {instanceState.environment.key_traits?.length > 0 && (
                          <div className="mt-3">
                            <span className="text-slate-500 text-sm">Key traits:</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {instanceState.environment.key_traits.map((trait, i) => (
                                <span key={i} className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                                  {trait}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
                {activeTab === 'metrics' && (
                  <MetricsPanel instanceState={instanceState} />
                )}
                {activeTab === 'history' && (
                  <HistoryPanel instanceId={selectedInstance} apiBase={API_BASE} />
                )}
                {activeTab === 'control' && (
                  <ControlExperiment
                    apiBase={API_BASE}
                    sessionId={sessionId}
                    apiKeyConfigured={apiKeyConfigured}
                    instanceId={selectedInstance}
                    onNurtureInteraction={handleInteraction}
                  />
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Brain className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <h2 className="text-xl font-semibold text-slate-400 mb-2">
                  No Instance Selected
                </h2>
                <p className="text-slate-500 mb-4">
                  Select an instance from the sidebar or create a new one
                </p>
                <button
                  onClick={createInstance}
                  className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-lg transition-colors mx-auto"
                >
                  <Plus className="w-4 h-4" />
                  Create Instance
                </button>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Error Toast */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg">
          {error}
          <button onClick={() => setError(null)} className="ml-4 font-bold">×</button>
        </div>
      )}
    </div>
  )
}

export default App
