import { useEffect, useRef } from 'react'

interface ChartData {
  timestamp: number
  value: number
}

interface DeviceChartProps {
  title: string
  data: ChartData[]
  unit?: string
  color?: string
  height?: number
}

export function DeviceChart({ 
  title, 
  data, 
  unit = '', 
  color = '#3b82f6',
  height = 200 
}: DeviceChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const width = rect.width
    const chartHeight = height - 40 // Leave space for labels
    const padding = { top: 10, right: 10, bottom: 30, left: 50 }

    // Clear canvas
    ctx.fillStyle = '#1e293b'
    ctx.fillRect(0, 0, width, height)

    // Calculate min/max
    const values = data.map(d => d.value)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const range = max - min || 1

    // Draw grid
    ctx.strokeStyle = '#334155'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartHeight / 4) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()

      // Y-axis labels
      const value = max - (range / 4) * i
      ctx.fillStyle = '#64748b'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'right'
      ctx.fillText(value.toFixed(1), padding.left - 5, y + 4)
    }

    // Draw X-axis labels
    const timeRange = data[data.length - 1].timestamp - data[0].timestamp
    for (let i = 0; i <= 4; i++) {
      const x = padding.left + ((width - padding.left - padding.right) / 4) * i
      const timestamp = data[0].timestamp + (timeRange / 4) * i
      const date = new Date(timestamp)
      const timeStr = `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`
      ctx.fillStyle = '#64748b'
      ctx.textAlign = 'center'
      ctx.fillText(timeStr, x, height - 10)
    }

    // Draw line
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    data.forEach((point, i) => {
      const x = padding.left + ((width - padding.left - padding.right) / (data.length - 1)) * i
      const y = padding.top + chartHeight - ((point.value - min) / range) * chartHeight
      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw fill with proper color conversion
    ctx.lineTo(padding.left + (width - padding.left - padding.right), padding.top + chartHeight)
    ctx.lineTo(padding.left, padding.top + chartHeight)
    ctx.closePath()
    const fillColor = (c: string): string => {
      if (c.startsWith('#')) {
        const hex = c.slice(1)
        if (hex.length === 6) {
          const r = parseInt(hex.slice(0, 2), 16)
          const g = parseInt(hex.slice(2, 4), 16)
          const b = parseInt(hex.slice(4, 6), 16)
          return `rgba(${r}, ${g}, ${b}, 0.3)`
        }
      }
      if (c.startsWith('rgb')) {
        return c.replace('rgb', 'rgba').replace(')', ', 0.3)')
      }
      return c
    }
    ctx.fillStyle = fillColor(color)
    ctx.fill()

    // Draw dots
    data.forEach((point, i) => {
      const x = padding.left + ((width - padding.left - padding.right) / (data.length - 1)) * i
      const y = padding.top + chartHeight - ((point.value - min) / range) * chartHeight
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
    })

    // Current value
    const currentValue = data[data.length - 1]?.value || 0
    ctx.fillStyle = color
    ctx.font = 'bold 14px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(`${currentValue.toFixed(1)}${unit}`, width - 80, 20)

  }, [data, color, height, unit])

  return (
    <div style={{ 
      background: '#1e293b', 
      borderRadius: '8px', 
      padding: '16px',
      color: '#e2e8f0'
    }}>
      <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>{title}</div>
      <canvas 
        ref={canvasRef} 
        style={{ 
          width: '100%', 
          height: `${height}px`,
          borderRadius: '4px'
        }} 
      />
    </div>
  )
}

interface StatsCardProps {
  title: string
  value: string | number
  unit?: string
  change?: number
  color?: string
}

export function StatsCard({ title, value, unit = '', change, color = '#3b82f6' }: StatsCardProps) {
  return (
    <div style={{ 
      background: '#1e293b', 
      borderRadius: '8px', 
      padding: '16px',
      color: '#e2e8f0'
    }}>
      <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '4px' }}>{title}</div>
      <div style={{ fontSize: '28px', fontWeight: 'bold', color }}>
        {value}<span style={{ fontSize: '14px', marginLeft: '4px' }}>{unit}</span>
      </div>
      {change !== undefined && (
        <div style={{ 
          fontSize: '12px', 
          color: change >= 0 ? '#10b981' : '#ef4444',
          marginTop: '4px'
        }}>
          {change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
        </div>
      )}
    </div>
  )
}