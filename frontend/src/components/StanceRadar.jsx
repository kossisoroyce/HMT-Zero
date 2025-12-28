import React from 'react'
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts'

const StanceRadar = ({ stance }) => {
  const data = Object.entries(stance).map(([key, value]) => ({
    dimension: key.charAt(0).toUpperCase() + key.slice(1),
    value: value * 100,
    fullMark: 100
  }))

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis 
            dataKey="dimension" 
            tick={{ fill: '#94a3b8', fontSize: 10 }}
          />
          <PolarRadiusAxis 
            angle={90} 
            domain={[0, 100]} 
            tick={{ fill: '#64748b', fontSize: 9 }}
            tickCount={5}
          />
          <Radar
            name="Stance"
            dataKey="value"
            stroke="#a855f7"
            fill="#a855f7"
            fillOpacity={0.3}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default StanceRadar
