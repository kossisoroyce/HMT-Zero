import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, Square, CheckCircle, Clock, AlertCircle, Beaker, ChevronDown, ChevronUp } from 'lucide-react'
import { TEST_PROTOCOL, getTotalPrompts, getAllPrompts } from '../testProtocol'

const AutomatedTestRunner = ({ 
  instanceId, 
  apiBase, 
  sessionId, 
  apiKeyConfigured,
  onInteraction,
  onComplete 
}) => {
  const [isRunning, setIsRunning] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0)
  const [results, setResults] = useState([])
  const [currentExperiment, setCurrentExperiment] = useState(null)
  const [showDetails, setShowDetails] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(false)
  const pauseRef = useRef(false)

  const allPrompts = getAllPrompts()
  const totalPrompts = getTotalPrompts()

  const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

  const runTest = async () => {
    if (!apiKeyConfigured) {
      setError('Please configure your OpenRouter API key first')
      return
    }

    setIsRunning(true)
    setIsPaused(false)
    setError(null)
    abortRef.current = false
    pauseRef.current = false

    for (let i = currentPromptIndex; i < allPrompts.length; i++) {
      if (abortRef.current) break

      // Check for pause
      while (pauseRef.current && !abortRef.current) {
        await delay(100)
      }
      if (abortRef.current) break

      const promptData = allPrompts[i]
      setCurrentPromptIndex(i)
      setCurrentExperiment(promptData.experimentName)

      try {
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
          const data = await res.json()
          throw new Error(data.detail || 'Interaction failed')
        }

        const data = await res.json()
        
        // Store result
        setResults(prev => [...prev, {
          ...promptData,
          response: data.response,
          metadata: data.metadata,
          state: data.state
        }])

        // Notify parent
        if (onInteraction) {
          onInteraction(data)
        }

        // Small delay between requests to avoid rate limiting
        if (i < allPrompts.length - 1) {
          await delay(1500)
        }

      } catch (err) {
        setError(`Error at prompt ${i + 1}: ${err.message}`)
        setIsRunning(false)
        return
      }
    }

    setIsRunning(false)
    setCurrentPromptIndex(0)
    if (onComplete && !abortRef.current) {
      onComplete(results)
    }
  }

  const pauseTest = () => {
    pauseRef.current = true
    setIsPaused(true)
  }

  const resumeTest = () => {
    pauseRef.current = false
    setIsPaused(false)
  }

  const stopTest = () => {
    abortRef.current = true
    pauseRef.current = false
    setIsRunning(false)
    setIsPaused(false)
  }

  const resetTest = () => {
    setCurrentPromptIndex(0)
    setResults([])
    setCurrentExperiment(null)
    setError(null)
  }

  const progress = results.length / totalPrompts * 100
  const evaluatedCount = results.filter(r => r.metadata?.was_evaluated).length

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Beaker className="w-5 h-5 text-purple-400" />
            <div>
              <h3 className="font-semibold text-white">Automated Test Protocol</h3>
              <p className="text-xs text-slate-400">{TEST_PROTOCOL.experiments.length} experiments, {totalPrompts} total prompts</p>
            </div>
          </div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-slate-400 hover:text-slate-200"
          >
            {showDetails ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Experiment List (collapsible) */}
      {showDetails && (
        <div className="p-4 border-b border-slate-700 max-h-48 overflow-y-auto">
          <div className="space-y-2">
            {TEST_PROTOCOL.experiments.map((exp, idx) => {
              const expResults = results.filter(r => r.experimentId === exp.id)
              const isDone = expResults.length === exp.prompts.length
              const isActive = currentExperiment === exp.name && isRunning
              
              return (
                <div 
                  key={exp.id}
                  className={`flex items-center justify-between text-sm p-2 rounded ${
                    isActive ? 'bg-purple-900/30 border border-purple-500/50' :
                    isDone ? 'bg-emerald-900/20' : 'bg-slate-800/50'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {isDone ? (
                      <CheckCircle className="w-4 h-4 text-emerald-400" />
                    ) : isActive ? (
                      <Clock className="w-4 h-4 text-purple-400 animate-pulse" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border border-slate-600" />
                    )}
                    <span className={isDone ? 'text-emerald-400' : isActive ? 'text-purple-300' : 'text-slate-400'}>
                      {exp.name}
                    </span>
                  </div>
                  <span className="text-xs text-slate-500">
                    {expResults.length}/{exp.prompts.length}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Progress */}
      <div className="p-4 space-y-3">
        {/* Progress bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Progress: {results.length}/{totalPrompts}</span>
            <span>{progress.toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-purple-500 to-emerald-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Stats */}
        {results.length > 0 && (
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className="bg-slate-800 rounded p-2">
              <div className="text-emerald-400 font-semibold">{evaluatedCount}</div>
              <div className="text-slate-500">Evaluated</div>
            </div>
            <div className="bg-slate-800 rounded p-2">
              <div className="text-amber-400 font-semibold">{results.length - evaluatedCount}</div>
              <div className="text-slate-500">Skipped</div>
            </div>
            <div className="bg-slate-800 rounded p-2">
              <div className="text-purple-400 font-semibold">
                {results.length > 0 ? (results.reduce((sum, r) => sum + (r.metadata?.significance_score || 0), 0) / results.length * 100).toFixed(0) : 0}%
              </div>
              <div className="text-slate-500">Avg Sig.</div>
            </div>
          </div>
        )}

        {/* Current prompt */}
        {isRunning && currentExperiment && (
          <div className="bg-slate-800 rounded p-3">
            <div className="text-xs text-slate-500 mb-1">Currently running:</div>
            <div className="text-sm text-purple-300">{currentExperiment}</div>
            <div className="text-xs text-slate-400 mt-1 truncate">
              "{allPrompts[currentPromptIndex]?.prompt}"
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 rounded p-3">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-2">
          {!isRunning ? (
            <>
              <button
                onClick={runTest}
                disabled={!apiKeyConfigured}
                className="flex-1 flex items-center justify-center gap-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:text-slate-500 text-white py-2.5 px-4 rounded-lg transition-colors font-medium"
              >
                <Play className="w-4 h-4" />
                {results.length > 0 && results.length < totalPrompts ? 'Resume Test' : 'Run Full Test'}
              </button>
              {results.length > 0 && (
                <button
                  onClick={resetTest}
                  className="px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
                >
                  Reset
                </button>
              )}
            </>
          ) : (
            <>
              {isPaused ? (
                <button
                  onClick={resumeTest}
                  className="flex-1 flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white py-2.5 px-4 rounded-lg transition-colors font-medium"
                >
                  <Play className="w-4 h-4" />
                  Resume
                </button>
              ) : (
                <button
                  onClick={pauseTest}
                  className="flex-1 flex items-center justify-center gap-2 bg-amber-600 hover:bg-amber-700 text-white py-2.5 px-4 rounded-lg transition-colors font-medium"
                >
                  <Pause className="w-4 h-4" />
                  Pause
                </button>
              )}
              <button
                onClick={stopTest}
                className="px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                <Square className="w-4 h-4" />
              </button>
            </>
          )}
        </div>

        {/* Completion message */}
        {results.length === totalPrompts && !isRunning && (
          <div className="bg-emerald-900/20 border border-emerald-600/50 rounded-lg p-3 text-center">
            <CheckCircle className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
            <div className="text-emerald-400 font-medium">Test Complete!</div>
            <div className="text-xs text-slate-400 mt-1">
              Click "Export JSON" above to download results for analysis
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default AutomatedTestRunner
