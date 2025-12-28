import React, { useState, useEffect } from 'react'
import { Play, Beaker, MessageSquare, Send, Download, RefreshCw, Server } from 'lucide-react'
import { TEST_PROTOCOL, getAllPrompts } from '../testProtocol'

const CONDITIONS = [
  { id: 'raw', name: 'Control A: Raw Model', description: 'No system prompt' },
  { id: 'static_prompt', name: 'Control B: Static Prompt', description: 'Best-case prompt engineering' },
  { id: 'nurture', name: 'Experimental: Nurture Layer', description: 'Dynamic character formation' }
]

const MODEL_PROVIDERS = [
  { id: 'openai', name: 'OpenAI GPT-4o', description: 'Requires API key' },
  { id: 'openrouter', name: 'OpenRouter', description: 'Mistral 7B via API - fast' },
  { id: 'ollama', name: 'Ollama (Local)', description: 'Local models - slow' }
]

const OPENROUTER_MODELS = [
  { id: 'mistralai/mistral-7b-instruct:free', name: 'Mistral 7B Instruct (FREE)', description: 'Free tier - recommended' },
  { id: 'mistral-7b', name: 'Mistral 7B Instruct', description: 'Fast, malleable' },
  { id: 'mistral-small', name: 'Mistral Small', description: 'Better quality' },
  { id: 'llama-3-8b', name: 'Llama 3 8B', description: 'Meta model' },
]

const ControlExperiment = ({ apiBase, sessionId, apiKeyConfigured, instanceId, onNurtureInteraction }) => {
  const [selectedCondition, setSelectedCondition] = useState('raw')
  const [selectedProvider, setSelectedProvider] = useState('openrouter')
  const [ollamaStatus, setOllamaStatus] = useState({ available: false, models: [] })
  const [selectedModel, setSelectedModel] = useState('mistralai/mistral-7b-instruct:free')
  const [openrouterApiKey, setOpenrouterApiKey] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState({ raw: [], static_prompt: [], nurture: [] })
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0)
  const [conversationHistory, setConversationHistory] = useState({ raw: [], static_prompt: [] })

  // Check Ollama status on mount
  useEffect(() => {
    checkOllamaStatus()
  }, [])

  const checkOllamaStatus = async () => {
    try {
      const res = await fetch(`${apiBase}/ollama/status`)
      const data = await res.json()
      setOllamaStatus(data)
      if (data.models?.length > 0) {
        // Prefer mistral if available
        const mistral = data.models.find(m => m.includes('mistral'))
        setOllamaModel(mistral || data.models[0])
      }
    } catch (err) {
      setOllamaStatus({ available: false, models: [] })
    }
  }
  
  const allPrompts = getAllPrompts()
  
  const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

  const runSingleCondition = async (condition) => {
    setIsRunning(true)
    const conditionResults = []
    const history = []
    
    for (let i = 0; i < allPrompts.length; i++) {
      setCurrentPromptIndex(i)
      const promptData = allPrompts[i]
      
      try {
        let response, data
        
        if (condition === 'nurture') {
          // Use the Nurture Layer endpoint
          const res = await fetch(`${apiBase}/interact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              instance_id: instanceId,
              user_input: promptData.prompt,
              session_id: sessionId
            })
          })
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`)
          }
          data = await res.json()
          response = data.response
          
          // Only call callback if we have valid data
          if (onNurtureInteraction && data && data.state) {
            onNurtureInteraction(data)
          }
        } else {
          // Use control endpoint
          const res = await fetch(`${apiBase}/control/interact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_input: promptData.prompt,
              session_id: sessionId,
              condition: condition,
              conversation_history: history,
              model_provider: selectedProvider,
              model_name: selectedModel,
              openrouter_api_key: selectedProvider === 'openrouter' ? openrouterApiKey : null
            })
          })
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`)
          }
          data = await res.json()
          response = data.response
          
          // Update history for next request
          history.push({ role: 'user', content: promptData.prompt })
          history.push({ role: 'assistant', content: response })
        }
        
        conditionResults.push({
          ...promptData,
          response: response,
          condition: condition,
          metadata: condition === 'nurture' ? data.metadata : null
        })
        
        // Delay between requests
        if (i < allPrompts.length - 1) {
          await delay(1500)
        }
        
      } catch (err) {
        console.error(`Error at prompt ${i}:`, err)
        conditionResults.push({
          ...promptData,
          response: `ERROR: ${err.message}`,
          condition: condition,
          error: true
        })
      }
    }
    
    setResults(prev => ({ ...prev, [condition]: conditionResults }))
    setIsRunning(false)
    setCurrentPromptIndex(0)
    
    return conditionResults
  }

  const exportResults = () => {
    // Build plotting-friendly trajectory data
    const buildTrajectory = (conditionResults, conditionName) => {
      return {
        condition: conditionName,
        interaction_numbers: conditionResults.map((_, i) => i + 1),
        experiments: conditionResults.map(r => r.experimentName),
        prompts: conditionResults.map(r => r.prompt),
        responses: conditionResults.map(r => r.response),
        response_lengths: conditionResults.map(r => r.response?.length || 0),
        // Nurture-specific metrics (null for control conditions)
        significance_scores: conditionResults.map(r => r.metadata?.significance_score || null),
        was_evaluated: conditionResults.map(r => r.metadata?.was_evaluated || null),
        delta_magnitudes: conditionResults.map(r => r.metadata?.delta_magnitude || null),
      }
    }

    const modelName = selectedProvider === 'openrouter' ? `openrouter/${selectedModel}` :
                      selectedProvider === 'ollama' ? `ollama/${selectedModel}` : 'openai/gpt-4o'
    
    const exportData = {
      export_version: '2.1',
      exported_at: new Date().toISOString(),
      experiment_type: 'control_comparison',
      model_provider: selectedProvider,
      model_name: modelName,
      total_prompts: allPrompts.length,
      
      // Summary stats
      summary: {
        raw: { total: results.raw.length, complete: results.raw.length === allPrompts.length, model: modelName },
        static_prompt: { total: results.static_prompt.length, complete: results.static_prompt.length === allPrompts.length, model: modelName },
        nurture: { 
          total: results.nurture.length, 
          complete: results.nurture.length === allPrompts.length,
          evaluated_count: results.nurture.filter(r => r.metadata?.was_evaluated).length,
          avg_significance: results.nurture.length > 0 
            ? results.nurture.reduce((sum, r) => sum + (r.metadata?.significance_score || 0), 0) / results.nurture.length 
            : 0
        }
      },

      // Plotting-friendly trajectories
      trajectories: {
        raw: buildTrajectory(results.raw, 'Control A: Raw GPT-4o'),
        static_prompt: buildTrajectory(results.static_prompt, 'Control B: Static Prompt'),
        nurture: buildTrajectory(results.nurture, 'Experimental: Nurture Layer')
      },

      // Full results for detailed analysis
      conditions: {
        raw: {
          name: 'Control A: Raw GPT-4o',
          description: 'No system prompt',
          results: results.raw
        },
        static_prompt: {
          name: 'Control B: Static Prompt',
          description: 'Best-case prompt engineering',
          results: results.static_prompt
        },
        nurture: {
          name: 'Experimental: Nurture Layer',
          description: 'Dynamic character formation',
          results: results.nurture
        }
      }
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `control-experiment-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const resetResults = () => {
    setResults({ raw: [], static_prompt: [], nurture: [] })
    setConversationHistory({ raw: [], static_prompt: [] })
  }

  const totalResults = results.raw.length + results.static_prompt.length + results.nurture.length
  const hasResults = totalResults > 0

  return (
    <div className="p-6 space-y-6 overflow-y-auto max-h-full">
      <div className="bg-purple-900/20 border border-purple-600/50 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-2">
          <Beaker className="w-5 h-5 text-purple-400" />
          <h3 className="text-sm font-semibold text-purple-400">Control Experiment Mode</h3>
        </div>
        <p className="text-sm text-slate-400">
          Run the same test protocol across three conditions to scientifically compare 
          Nurture Layer against baseline approaches.
        </p>
      </div>

      {/* Model Provider Selection */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium text-slate-300">Model Provider</h4>
          <button
            onClick={checkOllamaStatus}
            className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
          >
            <RefreshCw className="w-3 h-3" />
            Refresh
          </button>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {MODEL_PROVIDERS.map(provider => (
            <button
              key={provider.id}
              onClick={() => setSelectedProvider(provider.id)}
              disabled={isRunning || (provider.id === 'ollama' && !ollamaStatus.available)}
              className={`p-3 rounded-lg border text-left transition-colors ${
                selectedProvider === provider.id
                  ? 'border-cyan-500 bg-cyan-900/30'
                  : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
              } ${(provider.id === 'ollama' && !ollamaStatus.available) ? 'opacity-50' : ''}`}
            >
              <div className="flex items-center gap-2">
                <Server className={`w-4 h-4 ${
                  provider.id === 'ollama' ? 'text-orange-400' : 
                  provider.id === 'openrouter' ? 'text-purple-400' : 'text-green-400'
                }`} />
                <div>
                  <div className="text-sm font-medium text-slate-200">{provider.name}</div>
                  <div className="text-xs text-slate-500">{provider.description}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
        
        {/* OpenRouter Config */}
        {selectedProvider === 'openrouter' && (
          <div className="space-y-2">
            <input
              type="password"
              placeholder="OpenRouter API Key"
              value={openrouterApiKey}
              onChange={(e) => setOpenrouterApiKey(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder-slate-500"
            />
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200"
            >
              {OPENROUTER_MODELS.map(model => (
                <option key={model.id} value={model.id}>{model.name} - {model.description}</option>
              ))}
            </select>
            {!openrouterApiKey && (
              <p className="text-xs text-amber-400">
                Get API key from <a href="https://openrouter.ai/keys" target="_blank" rel="noopener" className="underline">openrouter.ai/keys</a>
              </p>
            )}
          </div>
        )}
        
        {/* Ollama Config */}
        {selectedProvider === 'ollama' && ollamaStatus.available && ollamaStatus.models.length > 0 && (
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200"
          >
            {ollamaStatus.models.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        )}
        {selectedProvider === 'ollama' && !ollamaStatus.available && (
          <p className="text-xs text-amber-400">
            Ollama not detected. Run: <code className="bg-slate-800 px-1 rounded">ollama serve</code> then <code className="bg-slate-800 px-1 rounded">ollama pull mistral</code>
          </p>
        )}
      </div>

      {/* Condition Selection */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-slate-300">Select Condition to Run</h4>
        <div className="grid grid-cols-1 gap-2">
          {CONDITIONS.map(cond => (
            <button
              key={cond.id}
              onClick={() => setSelectedCondition(cond.id)}
              disabled={isRunning || (cond.id === 'nurture' && !instanceId)}
              className={`p-3 rounded-lg border text-left transition-colors ${
                selectedCondition === cond.id
                  ? 'border-purple-500 bg-purple-900/30'
                  : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
              } ${(cond.id === 'nurture' && !instanceId) ? 'opacity-50' : ''}`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-slate-200">{cond.name}</div>
                  <div className="text-xs text-slate-500">{cond.description}</div>
                </div>
                <div className="text-xs text-slate-500">
                  {results[cond.id].length}/{allPrompts.length}
                </div>
              </div>
            </button>
          ))}
        </div>
        {selectedCondition === 'nurture' && !instanceId && (
          <p className="text-xs text-amber-400">Create an instance first to run Nurture Layer condition</p>
        )}
      </div>

      {/* Progress */}
      {isRunning && (
        <div className="bg-slate-800 rounded-lg p-4">
          <div className="flex justify-between text-sm text-slate-400 mb-2">
            <span>Running: {CONDITIONS.find(c => c.id === selectedCondition)?.name}</span>
            <span>{currentPromptIndex + 1}/{allPrompts.length}</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-purple-500 transition-all duration-300"
              style={{ width: `${((currentPromptIndex + 1) / allPrompts.length) * 100}%` }}
            />
          </div>
          <div className="text-xs text-slate-500 mt-2 truncate">
            "{allPrompts[currentPromptIndex]?.prompt}"
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={() => runSingleCondition(selectedCondition)}
          disabled={
            isRunning || 
            (selectedCondition === 'nurture' && !instanceId) ||
            (selectedCondition !== 'nurture' && selectedProvider === 'openai' && !apiKeyConfigured) ||
            (selectedCondition !== 'nurture' && selectedProvider === 'openrouter' && !openrouterApiKey) ||
            (selectedCondition !== 'nurture' && selectedProvider === 'ollama' && !ollamaStatus.available)
          }
          className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:text-slate-500 text-white py-2.5 px-4 rounded-lg transition-colors font-medium"
        >
          <Play className="w-4 h-4" />
          Run {CONDITIONS.find(c => c.id === selectedCondition)?.name.split(':')[0]}
          {selectedCondition !== 'nurture' && ` (${
            selectedProvider === 'openrouter' ? selectedModel :
            selectedProvider === 'ollama' ? selectedModel : 'GPT-4o'
          })`}
        </button>
        
        {hasResults && (
          <>
            <button
              onClick={exportResults}
              className="flex items-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={resetResults}
              className="flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </>
        )}
      </div>

      {/* Results Summary */}
      {hasResults && (
        <div className="bg-slate-800 rounded-lg p-4">
          <h4 className="text-sm font-medium text-slate-300 mb-3">Results Summary</h4>
          <div className="grid grid-cols-3 gap-4 text-center text-sm">
            {CONDITIONS.map(cond => (
              <div key={cond.id} className="bg-slate-700/50 rounded p-3">
                <div className="text-xs text-slate-500 mb-1">{cond.name.split(':')[0]}</div>
                <div className={`text-lg font-semibold ${
                  results[cond.id].length === allPrompts.length ? 'text-emerald-400' : 'text-slate-400'
                }`}>
                  {results[cond.id].length}/{allPrompts.length}
                </div>
              </div>
            ))}
          </div>
          
          {/* Comparison hint */}
          {results.raw.length === allPrompts.length && 
           results.static_prompt.length === allPrompts.length && 
           results.nurture.length === allPrompts.length && (
            <div className="mt-4 p-3 bg-emerald-900/20 border border-emerald-600/50 rounded text-sm text-emerald-400 text-center">
              ✓ All conditions complete! Export results for analysis.
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="text-xs text-slate-500 space-y-1">
        <p><strong>Recommended protocol:</strong></p>
        <ol className="list-decimal list-inside space-y-1 ml-2">
          <li>Run Control A (Raw GPT-4o)</li>
          <li>Run Control B (Static Prompt)</li>
          <li>Create fresh instance, then run Nurture Layer</li>
          <li>Export all results for comparison</li>
        </ol>
      </div>
    </div>
  )
}

export default ControlExperiment
