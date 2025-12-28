import React, { useState, useEffect } from 'react'
import { Clock, Zap, CheckCircle, XCircle } from 'lucide-react'

const HistoryPanel = ({ instanceId, apiBase }) => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchHistory()
  }, [instanceId])

  const fetchHistory = async () => {
    try {
      setLoading(true)
      const res = await fetch(`${apiBase}/instances/${instanceId}/history?limit=50`)
      const data = await res.json()
      setHistory(data.history || [])
    } catch (err) {
      console.error('Failed to fetch history:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-slate-500">Loading history...</div>
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500">
        <Clock className="w-12 h-12 mb-4 opacity-50" />
        <p>No interaction history yet</p>
        <p className="text-sm">Start interacting to build history</p>
      </div>
    )
  }

  return (
    <div className="p-6 overflow-y-auto h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-300">Interaction History</h3>
        <span className="text-sm text-slate-500">{history.length} interactions</span>
      </div>

      <div className="space-y-4">
        {history.slice().reverse().map((item, index) => (
          <div key={index} className="bg-slate-800 rounded-lg p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-slate-500" />
                <span className="text-xs text-slate-500">
                  {new Date(item.timestamp).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {item.metadata?.was_evaluated ? (
                  <span className="flex items-center gap-1 text-xs text-emerald-400">
                    <CheckCircle className="w-3 h-3" />
                    Evaluated
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-xs text-slate-500">
                    <XCircle className="w-3 h-3" />
                    Skipped
                  </span>
                )}
                {item.metadata?.significance_score && (
                  <span className="flex items-center gap-1 text-xs text-amber-400">
                    <Zap className="w-3 h-3" />
                    {(item.metadata.significance_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <div className="text-xs text-slate-500 mb-1">User</div>
                <div className="text-sm text-slate-300 bg-slate-700/50 rounded p-2">
                  {item.user_input}
                </div>
              </div>
              
              <div>
                <div className="text-xs text-slate-500 mb-1">Assistant</div>
                <div className="text-sm text-slate-400 bg-slate-700/30 rounded p-2">
                  {item.assistant_response}
                </div>
              </div>
            </div>

            {item.metadata && (
              <div className="mt-3 pt-3 border-t border-slate-700 flex flex-wrap gap-3 text-xs">
                {item.metadata.delta_magnitude > 0 && (
                  <span className="text-slate-500">
                    Delta: {item.metadata.delta_magnitude.toFixed(4)}
                  </span>
                )}
                {item.metadata.shock_detected && (
                  <span className="text-amber-400">Shock detected</span>
                )}
                {item.metadata.phase_before !== item.metadata.phase_after && (
                  <span className="text-purple-400">
                    Phase: {item.metadata.phase_before} → {item.metadata.phase_after}
                  </span>
                )}
                {item.metadata.simulated && (
                  <span className="text-slate-600">Simulated</span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default HistoryPanel
