import React from 'react'

const phaseConfig = {
  rapid_formation: {
    label: 'Rapid Formation',
    color: 'bg-amber-500',
    textColor: 'text-amber-400',
    description: 'Character highly malleable'
  },
  consolidation: {
    label: 'Consolidation',
    color: 'bg-blue-500',
    textColor: 'text-blue-400',
    description: 'Patterns forming'
  },
  stabilization: {
    label: 'Stabilization',
    color: 'bg-emerald-500',
    textColor: 'text-emerald-400',
    description: 'Character solidifying'
  },
  maturity: {
    label: 'Maturity',
    color: 'bg-purple-500',
    textColor: 'text-purple-400',
    description: 'Character established'
  }
}

const PhaseIndicator = ({ phase }) => {
  const config = phaseConfig[phase] || phaseConfig.rapid_formation

  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full ${config.color}`}></span>
      <span className={`text-sm font-medium ${config.textColor}`}>
        {config.label}
      </span>
    </div>
  )
}

export default PhaseIndicator
