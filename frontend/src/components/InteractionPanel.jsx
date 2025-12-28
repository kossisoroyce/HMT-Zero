import React, { useState } from 'react'
import { Send, Zap, AlertTriangle, CheckCircle, Key, Eye, EyeOff, MessageSquare, Beaker } from 'lucide-react'
import AutomatedTestRunner from './AutomatedTestRunner'

const InteractionPanel = ({ instanceId, instanceState, onInteraction, apiBase, sessionId, apiKeyConfigured, onApiKeyChange }) => {
  const [userInput, setUserInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastResult, setLastResult] = useState(null)
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [apiKeyLoading, setApiKeyLoading] = useState(false)
  const [apiKeyError, setApiKeyError] = useState(null)
  const [mode, setMode] = useState('manual') // 'manual' or 'automated'

  const handleSetApiKey = async () => {
    if (!apiKey.trim()) return
    
    setApiKeyLoading(true)
    setApiKeyError(null)
    
    try {
      const res = await fetch(`${apiBase}/api-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey, session_id: sessionId })
      })
      
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to set API key')
      }
      
      onApiKeyChange(true)
      setApiKey('')
    } catch (err) {
      setApiKeyError(err.message)
    } finally {
      setApiKeyLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!userInput.trim() || !apiKeyConfigured) return

    setLoading(true)
    try {
      const res = await fetch(`${apiBase}/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instance_id: instanceId,
          user_input: userInput,
          session_id: sessionId
        })
      })
      
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Interaction failed')
      }
      
      const data = await res.json()
      setLastResult(data)
      onInteraction(data)
      setUserInput('')
    } catch (err) {
      console.error('Interaction failed:', err)
      setLastResult({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* API Key Configuration */}
      {!apiKeyConfigured ? (
        <div className="bg-amber-900/20 border border-amber-600/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Key className="w-5 h-5 text-amber-400" />
            <h3 className="text-sm font-semibold text-amber-400">OpenRouter API Key Required</h3>
          </div>
          <p className="text-sm text-slate-400 mb-4">
            Enter your OpenRouter API key to enable Mistral 7B interactions. Get one free at openrouter.ai/keys
          </p>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-amber-500 pr-10"
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              >
                {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <button
              onClick={handleSetApiKey}
              disabled={apiKeyLoading || !apiKey.trim()}
              className="px-4 py-2 bg-amber-600 hover:bg-amber-700 disabled:bg-slate-700 text-white rounded-lg font-medium transition-colors"
            >
              {apiKeyLoading ? 'Validating...' : 'Save Key'}
            </button>
          </div>
          {apiKeyError && (
            <p className="text-sm text-red-400 mt-2">{apiKeyError}</p>
          )}
        </div>
      ) : (
        <div className="bg-emerald-900/20 border border-emerald-600/50 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span className="text-sm text-emerald-400">OpenRouter API key configured (Mistral 7B)</span>
          </div>
          <button
            onClick={() => onApiKeyChange(false)}
            className="text-xs text-slate-500 hover:text-slate-300"
          >
            Change key
          </button>
        </div>
      )}

      {/* Mode Toggle */}
      <div className="flex gap-2 p-1 bg-slate-800 rounded-lg">
        <button
          onClick={() => setMode('manual')}
          className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            mode === 'manual'
              ? 'bg-purple-600 text-white'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <MessageSquare className="w-4 h-4" />
          Manual Chat
        </button>
        <button
          onClick={() => setMode('automated')}
          className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            mode === 'automated'
              ? 'bg-purple-600 text-white'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Beaker className="w-4 h-4" />
          Test Protocol
        </button>
      </div>

      {/* Automated Test Runner */}
      {mode === 'automated' && (
        <AutomatedTestRunner
          instanceId={instanceId}
          apiBase={apiBase}
          sessionId={sessionId}
          apiKeyConfigured={apiKeyConfigured}
          onInteraction={onInteraction}
        />
      )}

      {/* Manual Input Form */}
      {mode === 'manual' && (
      <>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Your Message
          </label>
          <textarea
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={apiKeyConfigured ? "Enter your message..." : "Configure API key first..."}
            disabled={!apiKeyConfigured}
            className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 resize-none disabled:opacity-50"
            rows={4}
          />
        </div>

        <button
          type="submit"
          disabled={loading || !userInput.trim() || !apiKeyConfigured}
          className="w-full flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:text-slate-500 text-white py-3 px-4 rounded-lg transition-colors font-medium"
        >
          {loading ? (
            'Processing with GPT-4o...'
          ) : (
            <>
              <Send className="w-4 h-4" />
              Send Message
            </>
          )}
        </button>
      </form>

      {/* Response Display */}
      {lastResult && !lastResult.error && (
        <div className="bg-slate-800 rounded-lg p-4 space-y-3">
          <h3 className="text-sm font-semibold text-slate-300">Response</h3>
          <p className="text-slate-200 whitespace-pre-wrap">{lastResult.response}</p>
          
          <div className="pt-3 border-t border-slate-700 grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Zap className={`w-4 h-4 ${lastResult.metadata.was_evaluated ? 'text-amber-400' : 'text-slate-500'}`} />
              <span className="text-slate-400">
                Significance: {(lastResult.metadata.significance_score * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="flex items-center gap-2">
              {lastResult.metadata.was_evaluated ? (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              ) : (
                <span className="w-4 h-4 rounded-full bg-slate-600" />
              )}
              <span className="text-slate-400">
                {lastResult.metadata.was_evaluated ? 'Evaluated' : 'Skipped'}
              </span>
            </div>

            {lastResult.metadata.shock_detected && (
              <div className="col-span-2 flex items-center gap-2 text-amber-400">
                <AlertTriangle className="w-4 h-4" />
                <span>Shock detected - plasticity increased</span>
              </div>
            )}

            {lastResult.metadata.phase_transition && (
              <div className="col-span-2 flex items-center gap-2 text-purple-400">
                <span>Phase transition: {lastResult.metadata.phase_before} → {lastResult.metadata.phase_after}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {lastResult?.error && (
        <div className="bg-red-900/20 border border-red-600/50 rounded-lg p-4">
          <p className="text-sm text-red-400">{lastResult.error}</p>
        </div>
      )}
      </>
      )}

      {/* Current Threshold Info */}
      <div className="bg-slate-800/50 rounded-lg p-3 text-sm">
        <div className="flex justify-between text-slate-400">
          <span>Current evaluation threshold:</span>
          <span className="text-slate-300">{(instanceState.current_threshold * 100).toFixed(1)}%</span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Threshold is dynamic: lower when plastic (catches more), higher when stable
        </p>
      </div>
    </div>
  )
}

export default InteractionPanel
