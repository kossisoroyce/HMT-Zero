import React, { useState, useEffect } from 'react'
import { Brain, MessageCircle, HelpCircle, CheckSquare, Sparkles, RefreshCw, Send } from 'lucide-react'

const API_BASE = '/api'

function ExperientialPanel({ instanceId, sessionId, apiKeyConfigured, openrouterApiKey }) {
  const [experientialState, setExperientialState] = useState(null)
  const [facts, setFacts] = useState([])
  const [questions, setQuestions] = useState([])
  const [commitments, setCommitments] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sessionActive, setSessionActive] = useState(false)
  
  // Integrated interaction state
  const [message, setMessage] = useState('')
  const [conversation, setConversation] = useState([])
  const [interacting, setInteracting] = useState(false)

  useEffect(() => {
    if (sessionActive) {
      fetchExperientialState()
    }
  }, [sessionId, sessionActive])

  const startSession = async () => {
    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/experience/session?instance_id=${instanceId}&session_id=${sessionId}`, {
        method: 'POST'
      })
      if (!res.ok) throw new Error('Failed to start session')
      setSessionActive(true)
      await fetchExperientialState()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const endSession = async () => {
    try {
      setLoading(true)
      const res = await fetch(`${API_BASE}/experience/session/${sessionId}`, {
        method: 'DELETE'
      })
      const data = await res.json()
      setSessionActive(false)
      setExperientialState(null)
      setFacts([])
      setQuestions([])
      setCommitments([])
      setConversation([])
      return data
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchExperientialState = async () => {
    try {
      const [stateRes, factsRes, questionsRes, commitmentsRes] = await Promise.all([
        fetch(`${API_BASE}/experience/session/${sessionId}`),
        fetch(`${API_BASE}/experience/facts/${sessionId}`),
        fetch(`${API_BASE}/experience/questions/${sessionId}`),
        fetch(`${API_BASE}/experience/commitments/${sessionId}`)
      ])
      
      if (stateRes.ok) {
        setExperientialState(await stateRes.json())
      }
      if (factsRes.ok) {
        const factsData = await factsRes.json()
        setFacts(factsData.facts || [])
      }
      if (questionsRes.ok) {
        const questionsData = await questionsRes.json()
        setQuestions(questionsData.questions || [])
      }
      if (commitmentsRes.ok) {
        const commitmentsData = await commitmentsRes.json()
        setCommitments(commitmentsData.commitments || [])
      }
    } catch (err) {
      console.error('Failed to fetch experiential state:', err)
    }
  }

  const sendIntegratedMessage = async () => {
    if (!message.trim() || !apiKeyConfigured) return
    
    const userMessage = message.trim()
    setMessage('')
    setConversation(prev => [...prev, { role: 'user', content: userMessage }])
    setInteracting(true)
    
    try {
      const res = await fetch(`${API_BASE}/integrated/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instance_id: instanceId,
          session_id: sessionId,
          user_input: userMessage,
          openrouter_api_key: openrouterApiKey,
          model_name: 'mistralai/mistral-7b-instruct:free'
        })
      })
      
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Request failed')
      }
      
      const data = await res.json()
      setConversation(prev => [...prev, { role: 'assistant', content: data.response }])
      setExperientialState(data.experiential_state)
      
      // Refresh working memory
      await fetchExperientialState()
    } catch (err) {
      setError(err.message)
      setConversation(prev => [...prev, { role: 'error', content: err.message }])
    } finally {
      setInteracting(false)
    }
  }

  if (!sessionActive) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <Sparkles className="w-16 h-16 text-cyan-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Experiential Layer</h2>
          <p className="text-slate-400 mb-6">
            The Experiential Layer tracks session context, extracts salient facts, 
            monitors open questions, and remembers commitments. It integrates with 
            the Nurture Layer for full CACA stack processing.
          </p>
          <button
            onClick={startSession}
            disabled={loading || !instanceId}
            className="flex items-center gap-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-600 text-white py-3 px-6 rounded-lg transition-colors mx-auto"
          >
            <Brain className="w-5 h-5" />
            Start Experiential Session
          </button>
          {!instanceId && (
            <p className="text-amber-400 text-sm mt-4">Select a nurture instance first</p>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Session Header */}
        <div className="bg-slate-800/50 border-b border-slate-700 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              <div>
                <span className="text-white font-medium">Integrated Session</span>
                <span className="text-slate-400 text-sm ml-2">
                  (Nurture + Experience)
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchExperientialState}
                className="p-2 hover:bg-slate-700 rounded text-slate-400 hover:text-white"
                title="Refresh state"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
              <button
                onClick={endSession}
                className="text-sm text-red-400 hover:text-red-300 px-3 py-1 hover:bg-red-900/20 rounded"
              >
                End Session
              </button>
            </div>
          </div>
        </div>

        {/* Conversation */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {conversation.length === 0 && (
            <div className="text-center text-slate-500 py-8">
              <MessageCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Start a conversation to see integrated Nurture + Experience processing</p>
            </div>
          )}
          {conversation.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  msg.role === 'user'
                    ? 'bg-cyan-600 text-white'
                    : msg.role === 'error'
                    ? 'bg-red-900/50 text-red-200'
                    : 'bg-slate-700 text-slate-200'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}
          {interacting && (
            <div className="flex justify-start">
              <div className="bg-slate-700 text-slate-400 rounded-lg px-4 py-2">
                <span className="animate-pulse">Processing through CACA stack...</span>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-slate-700 p-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendIntegratedMessage()}
              placeholder="Type a message..."
              className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
              disabled={interacting}
            />
            <button
              onClick={sendIntegratedMessage}
              disabled={!message.trim() || interacting || !apiKeyConfigured}
              className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          {!apiKeyConfigured && (
            <p className="text-amber-400 text-sm mt-2">Configure OpenRouter API key first</p>
          )}
        </div>
      </div>

      {/* Right Sidebar - Experiential State */}
      <div className="w-80 border-l border-slate-700 bg-slate-800/30 overflow-y-auto">
        {/* Context Summary */}
        {experientialState && (
          <div className="p-4 border-b border-slate-700">
            <h3 className="text-sm font-semibold text-cyan-400 mb-3 flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Session State
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Interactions</span>
                <span className="text-white">{experientialState.interaction_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Topic</span>
                <span className="text-white truncate ml-2">{experientialState.topic_summary || '—'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Emotion</span>
                <span className={`${
                  experientialState.emotion_summary?.includes('positive') ? 'text-green-400' :
                  experientialState.emotion_summary?.includes('negative') ? 'text-red-400' :
                  'text-slate-300'
                }`}>
                  {experientialState.emotion_summary || 'neutral'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">User State</span>
                <span className="text-white">{experientialState.user_summary || '—'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Familiarity</span>
                <span className="text-white">{(experientialState.session_familiarity * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Salient Facts */}
        <div className="p-4 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-emerald-400 mb-3 flex items-center gap-2">
            <MessageCircle className="w-4 h-4" />
            Salient Facts ({facts.length})
          </h3>
          {facts.length === 0 ? (
            <p className="text-slate-500 text-sm">No facts extracted yet</p>
          ) : (
            <div className="space-y-2">
              {facts.slice(0, 5).map((fact, i) => (
                <div key={i} className="bg-slate-800 rounded p-2 text-sm">
                  <p className="text-slate-300">{fact.content}</p>
                  <div className="flex justify-between mt-1 text-xs text-slate-500">
                    <span>{fact.source}</span>
                    <span>salience: {(fact.salience_score * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Open Questions */}
        <div className="p-4 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-amber-400 mb-3 flex items-center gap-2">
            <HelpCircle className="w-4 h-4" />
            Open Questions ({questions.filter(q => !q.resolved).length})
          </h3>
          {questions.length === 0 ? (
            <p className="text-slate-500 text-sm">No questions tracked</p>
          ) : (
            <div className="space-y-2">
              {questions.slice(0, 5).map((q, i) => (
                <div key={i} className={`bg-slate-800 rounded p-2 text-sm ${q.resolved ? 'opacity-50' : ''}`}>
                  <p className="text-slate-300">{q.question}</p>
                  <div className="flex justify-between mt-1 text-xs">
                    <span className={q.resolved ? 'text-green-400' : 'text-amber-400'}>
                      {q.resolved ? '✓ resolved' : 'open'}
                    </span>
                    <span className="text-slate-500">attempts: {q.attempted_answers}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Commitments */}
        <div className="p-4">
          <h3 className="text-sm font-semibold text-purple-400 mb-3 flex items-center gap-2">
            <CheckSquare className="w-4 h-4" />
            Commitments ({commitments.filter(c => !c.fulfilled).length})
          </h3>
          {commitments.length === 0 ? (
            <p className="text-slate-500 text-sm">No commitments made</p>
          ) : (
            <div className="space-y-2">
              {commitments.slice(0, 5).map((c, i) => (
                <div key={i} className={`bg-slate-800 rounded p-2 text-sm ${c.fulfilled ? 'opacity-50' : ''}`}>
                  <p className="text-slate-300">{c.promise}</p>
                  <span className={`text-xs ${c.fulfilled ? 'text-green-400' : 'text-purple-400'}`}>
                    {c.fulfilled ? '✓ fulfilled' : 'active'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Context String Preview */}
        {experientialState?.context_string && (
          <div className="p-4 border-t border-slate-700">
            <h3 className="text-sm font-semibold text-slate-400 mb-2">Context Injection</h3>
            <pre className="text-xs text-slate-500 bg-slate-900 p-2 rounded overflow-x-auto whitespace-pre-wrap">
              {experientialState.context_string || '(empty)'}
            </pre>
          </div>
        )}
      </div>

      {/* Error Toast */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg z-50">
          {error}
          <button onClick={() => setError(null)} className="ml-4 font-bold">×</button>
        </div>
      )}
    </div>
  )
}

export default ExperientialPanel
