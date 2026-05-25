import { useMemo } from 'react'

export interface DemandForecastPoint {
  date: string
  actual?: number
  forecast: number
  lowerBound?: number
  upperBound?: number
}

export interface DemandChartProps {
  data: DemandForecastPoint[]
  title?: string
  showConfidenceInterval?: boolean
  showActual?: boolean
  height?: number
}

export function DemandChart({ 
  data, 
  title = '需求预测趋势',
  showConfidenceInterval = true,
  showActual = true,
  height = 300
}: DemandChartProps) {
  const chartData = useMemo(() => {
    if (!data.length) return { maxVal: 100, points: [] }
    
    const maxVal = Math.max(
      ...data.flatMap(d => [
        d.actual || 0, 
        d.forecast, 
        d.upperBound || d.forecast, 
        d.lowerBound || d.forecast
      ])
    )
    
    const points = data.map((d, i) => ({
      ...d,
      x: (i / Math.max(data.length - 1, 1)) * 100,
      y: 100 - (d.forecast / maxVal) * 80 - 10
    }))
    
    return { maxVal, points }
  }, [data])

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return `${date.getMonth() + 1}/${date.getDate()}`
  }

  // Generate SVG paths
  const forecastPath = useMemo(() => {
    if (chartData.points.length < 2) return ''
    return chartData.points.map((p, i) => 
      `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
    ).join(' ')
  }, [chartData.points])

  const upperBoundPath = useMemo(() => {
    if (!showConfidenceInterval || chartData.points.length < 2) return ''
    const upperPoints = chartData.points.map(p => ({
      x: p.x,
      y: 100 - ((p.upperBound || p.forecast) / chartData.maxVal) * 80 - 10
    }))
    return upperPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  }, [chartData.points, showConfidenceInterval])

  const lowerBoundPath = useMemo(() => {
    if (!showConfidenceInterval || chartData.points.length < 2) return ''
    const lowerPoints = chartData.points.map(p => ({
      x: p.x,
      y: 100 - ((p.lowerBound || p.forecast) / chartData.maxVal) * 80 - 10
    }))
    return lowerPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  }, [chartData.points, showConfidenceInterval])

  const actualPath = useMemo(() => {
    if (!showActual) return ''
    const actualPoints = chartData.points.filter(p => p.actual !== undefined)
    if (actualPoints.length < 2) return ''
    return actualPoints.map((p, i) => 
      `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
    ).join(' ')
  }, [chartData.points, showActual])

  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.9)',
      borderRadius: '12px',
      border: '1px solid #334155',
      color: '#f1f5f9',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '16px'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '16px'
      }}>
        <span style={{ fontWeight: 600, fontSize: '15px' }}>{title}</span>
        <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
          {showActual && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '12px', height: '3px', background: '#22c55e', borderRadius: '2px' }} />
              实际值
            </span>
          )}
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '3px', background: '#38bdf8', borderRadius: '2px' }} />
            预测值
          </span>
          {showConfidenceInterval && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '12px', height: '3px', background: 'rgba(56, 189, 248, 0.3)', borderRadius: '2px' }} />
              置信区间
            </span>
          )}
        </div>
      </div>

      {/* Chart area */}
      <div style={{ position: 'relative', height: `${height}px` }}>
        <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(y => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="100"
              y2={y}
              stroke="#334155"
              strokeWidth="0.2"
            />
          ))}
          
          {/* Confidence interval area */}
          {showConfidenceInterval && upperBoundPath && lowerBoundPath && (
            <path
              d={`${upperBoundPath} L ${chartData.points[chartData.points.length - 1]?.x || 100} ${chartData.points[chartData.points.length - 1]?.y || 10} L ${chartData.points[0]?.x || 0} ${chartData.points[0]?.y || 10} Z`}
              fill="rgba(56, 189, 248, 0.15)"
              stroke="none"
            />
          )}
          
          {/* Forecast line */}
          {forecastPath && (
            <path
              d={forecastPath}
              fill="none"
              stroke="#38bdf8"
              strokeWidth="0.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          
          {/* Actual line */}
          {actualPath && (
            <path
              d={actualPath}
              fill="none"
              stroke="#22c55e"
              strokeWidth="0.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray="2,1"
            />
          )}
          
          {/* Data points */}
          {chartData.points.map((p, i) => (
            <g key={i}>
              {/* Forecast point */}
              <circle
                cx={p.x}
                cy={p.y}
                r="1.2"
                fill="#0f172a"
                stroke="#38bdf8"
                strokeWidth="0.4"
              />
              
              {/* Actual point (if exists) */}
              {showActual && p.actual !== undefined && (
                <circle
                  cx={p.x}
                  cy={100 - (p.actual / chartData.maxVal) * 80 - 10}
                  r="1"
                  fill="#22c55e"
                  stroke="#0f172a"
                  strokeWidth="0.3"
                />
              )}
            </g>
          ))}
        </svg>
        
        {/* X-axis labels */}
        <div style={{
          position: 'absolute',
          bottom: '-24px',
          left: 0,
          right: 0,
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '10px',
          color: '#64748b'
        }}>
          {data.length > 0 && (
            <>
              <span>{formatDate(data[0]?.date || '')}</span>
              <span>{formatDate(data[Math.floor(data.length / 2)]?.date || '')}</span>
              <span>{formatDate(data[data.length - 1]?.date || '')}</span>
            </>
          )}
        </div>
        
        {/* Y-axis labels */}
        <div style={{
          position: 'absolute',
          left: '-32px',
          top: 0,
          bottom: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          fontSize: '10px',
          color: '#64748b'
        }}>
          <span>{Math.round(chartData.maxVal)}</span>
          <span>{Math.round(chartData.maxVal * 0.75)}</span>
          <span>{Math.round(chartData.maxVal * 0.5)}</span>
          <span>{Math.round(chartData.maxVal * 0.25)}</span>
          <span>0</span>
        </div>
      </div>

      {/* Stats summary */}
      <div style={{
        marginTop: '20px',
        paddingTop: '12px',
        borderTop: '1px solid #334155',
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#64748b', fontSize: '10px', marginBottom: '2px' }}>最新预测</div>
          <div style={{ fontSize: '16px', fontWeight: 600, color: '#38bdf8' }}>
            {data.length > 0 ? Math.round(data[data.length - 1].forecast) : '-'}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#64748b', fontSize: '10px', marginBottom: '2px' }}>平均预测</div>
          <div style={{ fontSize: '16px', fontWeight: 600, color: '#f1f5f9' }}>
            {data.length > 0 ? Math.round(data.reduce((sum, d) => sum + d.forecast, 0) / data.length) : '-'}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#64748b', fontSize: '10px', marginBottom: '2px' }}>趋势</div>
          <div style={{ 
            fontSize: '16px', 
            fontWeight: 600, 
            color: data.length > 1 && data[data.length - 1].forecast > data[0].forecast ? '#22c55e' : '#ef4444' 
          }}>
            {data.length > 1 
              ? (data[data.length - 1].forecast > data[0].forecast ? '↑ 上升' : '↓ 下降')
              : '-'
            }
          </div>
        </div>
      </div>
    </div>
  )
}

export type { DemandChartProps, DemandForecastPoint }