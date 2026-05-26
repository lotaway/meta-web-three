import { useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Colors, Spacing, FontSizes, BorderRadius } from './styles/constants'
import { formatDate } from './utils/format'

// ============ 常量提取 ============
const CHART_CONFIG = {
  gridY: [0, 25, 50, 75, 100] as const,
  yPaddingTop: 10,
  yScale: 80,
  strokeWidth: {
    grid: 0.2,
    forecast: 0.8,
    actual: 0.6,
    point: 0.4,
    pointActual: 0.3
  },
  pointRadius: {
    forecast: 1.2,
    actual: 1
  },
  dashArray: '2,1',
  legendLineWidth: '12px',
  legendLineHeight: '3px',
  legendLineRadius: '2px',
  legendGap: '6px',
  axisLabelBottom: {
    marginTop: '-24px'
  },
  axisLabelLeft: {
    marginLeft: '-32px'
  },
  svgViewBox: '0 0 100 100',
  preserveAspectRatio: 'none' as const
} as const

const CONFIDENCE_INTERVAL_COLORS = {
  fill: 'rgba(56, 189, 248, 0.15)',
  stroke: 'rgba(56, 189, 248, 0.3)'
} as const

const CHART_COLORS = {
  pointFill: '#0f172a',
  text: Colors.text,
  textMuted: Colors.textMuted
} as const

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

interface ChartStatsProps {
  data: DemandForecastPoint[]
}

// ============ 子组件：统计信息 ============
function ChartStats({ data }: ChartStatsProps) {
  if (data.length === 0) return null

  const latestForecast = data[data.length - 1].forecast
  const avgForecast = Math.round(data.reduce((sum, d) => sum + d.forecast, 0) / data.length)
  const trend = data.length > 1
    ? data[data.length - 1].forecast > data[0].forecast
    : false

  const trendText = trend ? '↑ 上升' : '↓ 下降'
  const trendColor = trend ? Colors.success : Colors.danger

  return (
    <div
      role="region"
      aria-label="统计信息"
      style={{
        marginTop: Spacing.xl,
        paddingTop: Spacing.lg,
        borderTop: `1px solid ${Colors.border}`,
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: Spacing.lg,
        textAlign: 'center'
      }}
    >
      <div>
        <div style={{ color: Colors.textMuted, fontSize: FontSizes.xs, marginBottom: '2px' }}>
          最新预测
        </div>
        <div style={{ fontSize: FontSizes.xxl, fontWeight: 600, color: Colors.primary }}>
          {Math.round(latestForecast)}
        </div>
      </div>
      <div>
        <div style={{ color: Colors.textMuted, fontSize: FontSizes.xs, marginBottom: '2px' }}>
          平均预测
        </div>
        <div style={{ fontSize: FontSizes.xxl, fontWeight: 600, color: Colors.text }}>
          {avgForecast}
        </div>
      </div>
      <div>
        <div style={{ color: Colors.textMuted, fontSize: FontSizes.xs, marginBottom: '2px' }}>
          趋势
        </div>
        <div style={{ fontSize: FontSizes.xxl, fontWeight: 600, color: trendColor }}>
          {trendText}
        </div>
      </div>
    </div>
  )
}

// ============ 子组件：图例 ============
function ChartLegend({ showActual, showConfidenceInterval }: {
  showActual: boolean
  showConfidenceInterval: boolean
}) {
  const legendItems = []
  if (showActual) legendItems.push({ color: Colors.success, label: '实际值' })
  legendItems.push({ color: Colors.primary, label: '预测值' })
  if (showConfidenceInterval) legendItems.push({ color: CONFIDENCE_INTERVAL_COLORS.stroke, label: '置信区间' })

  return (
    <div
      role="list"
      aria-label="图例"
      style={{ display: 'flex', gap: Spacing.xl, fontSize: FontSizes.sm }}
    >
      {legendItems.map(item => (
        <span
          key={item.label}
          role="listitem"
          style={{ display: 'flex', alignItems: 'center', gap: CHART_CONFIG.legendGap }}
        >
          <span
            style={{
              width: CHART_CONFIG.legendLineWidth,
              height: CHART_CONFIG.legendLineHeight,
              background: item.color,
              borderRadius: CHART_CONFIG.legendLineRadius
            }}
          />
          {item.label}
        </span>
      ))}
    </div>
  )
}

// ============ 子组件：图表头部 ============
function ChartHeader({ title, legend }: { title: string; legend: React.ReactNode }) {
  return (
    <div
      role="banner"
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.xl
      }}
    >
      <span style={{ fontWeight: 600, fontSize: FontSizes.xl }}>{title}</span>
      {legend}
    </div>
  )
}

// ============ 子组件：Y轴标签 ============
function YAxisLabels({ maxVal }: { maxVal: number }) {
  const labels = [
    Math.round(maxVal),
    Math.round(maxVal * 0.75),
    Math.round(maxVal * 0.5),
    Math.round(maxVal * 0.25),
    0
  ]

  return (
    <div
      role="list"
      aria-label="Y轴标签"
      style={{
        position: 'absolute',
        left: CHART_CONFIG.axisLabelLeft.marginLeft,
        top: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        fontSize: FontSizes.xs,
        color: Colors.textMuted
      }}
    >
      {labels.map((label, i) => (
        <span key={i} role="listitem">{label}</span>
      ))}
    </div>
  )
}

// ============ 子组件：X轴标签 ============
function XAxisLabels({ data }: { data: DemandForecastPoint[] }) {
  if (data.length === 0) return null

  const positions = [
    data[0]?.date || '',
    data[Math.floor(data.length / 2)]?.date || '',
    data[data.length - 1]?.date || ''
  ]

  return (
    <div
      role="list"
      aria-label="X轴标签"
      style={{
        position: 'absolute',
        bottom: CHART_CONFIG.axisLabelBottom.marginTop,
        left: 0,
        right: 0,
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: FontSizes.xs,
        color: Colors.textMuted
      }}
    >
      {positions.map((date, i) => (
        <span key={i} role="listitem">{formatDate(date)}</span>
      ))}
    </div>
  )
}

// ============ 子组件：网格线 ============
function GridLines() {
  return (
    <>
      {CHART_CONFIG.gridY.map(y => (
        <line
          key={y}
          x1="0"
          y1={y}
          x2="100"
          y2={y}
          stroke={Colors.border}
          strokeWidth={CHART_CONFIG.strokeWidth.grid}
        />
      ))}
    </>
  )
}

// ============ 子组件：置信区间 ============
function ConfidenceInterval({
  upper,
  lower,
  maxX
}: {
  upper: string
  lower: string
  maxX: number
}) {
  if (!upper || !lower) return null

  const path = `${upper} L ${maxX} ${CHART_CONFIG.yPaddingTop} L 0 ${CHART_CONFIG.yPaddingTop} Z`
  return (
    <path
      d={path}
      fill={CONFIDENCE_INTERVAL_COLORS.fill}
      stroke="none"
    />
  )
}

// ============ 子组件：预测曲线 ============
function ForecastPath({ d }: { d: string }) {
  if (!d) return null
  return (
    <path
      d={d}
      fill="none"
      stroke={Colors.primary}
      strokeWidth={CHART_CONFIG.strokeWidth.forecast}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  )
}

// ============ 子组件：实际值曲线 ============
function ActualPath({ d }: { d: string }) {
  if (!d) return null
  return (
    <path
      d={d}
      fill="none"
      stroke={Colors.success}
      strokeWidth={CHART_CONFIG.strokeWidth.actual}
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeDasharray={CHART_CONFIG.dashArray}
    />
  )
}

// ============ 子组件：数据点 ============
function DataPoints({
  points,
  showActual,
  maxVal
}: {
  points: Array<{ x: number; y: number; actual?: number; forecast: number }>
  showActual: boolean
  maxVal: number
}) {
  return (
    <>
      {points.map((p, i) => (
        <g key={i}>
          <circle
            cx={p.x}
            cy={p.y}
            r={CHART_CONFIG.pointRadius.forecast}
            fill={CHART_COLORS.pointFill}
            stroke={Colors.primary}
            strokeWidth={CHART_CONFIG.strokeWidth.point}
          />
          {showActual && p.actual !== undefined && (
            <circle
              cx={p.x}
              cy={100 - (p.actual / maxVal) * CHART_CONFIG.yScale - CHART_CONFIG.yPaddingTop}
              r={CHART_CONFIG.pointRadius.actual}
              fill={Colors.success}
              stroke={CHART_COLORS.pointFill}
              strokeWidth={CHART_CONFIG.strokeWidth.pointActual}
            />
          )}
        </g>
      ))}
    </>
  )
}

// ============ 主组件 ============
export function DemandChart({
  data,
  title = '需求预测趋势',
  showConfidenceInterval = true,
  showActual = true,
  height = 300
}: DemandChartProps) {
  // 计算图表数据
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
      y: 100 - (d.forecast / maxVal) * CHART_CONFIG.yScale - CHART_CONFIG.yPaddingTop
    }))

    return { maxVal, points }
  }, [data])

  // 计算路径
  const paths = useMemo(() => {
    if (chartData.points.length < 2) {
      return { forecast: '', upper: '', lower: '', actual: '' }
    }

    const forecast = chartData.points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
      .join(' ')

    const upper = showConfidenceInterval
      ? chartData.points
          .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${100 - ((p.upperBound || p.forecast) / chartData.maxVal) * CHART_CONFIG.yScale - CHART_CONFIG.yPaddingTop}`)
          .join(' ')
      : ''

    const lower = showConfidenceInterval
      ? chartData.points
          .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${100 - ((p.lowerBound || p.forecast) / chartData.maxVal) * CHART_CONFIG.yScale - CHART_CONFIG.yPaddingTop}`)
          .join(' ')
      : ''

    const actual = showActual
      ? chartData.points
          .filter(p => p.actual !== undefined)
          .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${100 - (p.actual! / chartData.maxVal) * CHART_CONFIG.yScale - CHART_CONFIG.yPaddingTop}`)
          .join(' ')
      : ''

    return { forecast, upper, lower, actual }
  }, [chartData, showConfidenceInterval, showActual])

  const legend = (
    <ChartLegend showActual={showActual} showConfidenceInterval={showConfidenceInterval} />
  )

  return (
    <ErrorBoundary>
      <Card style={{ padding: Spacing.xl }}>
        <ChartHeader title={title} legend={legend} />

        <div style={{ position: 'relative', height: `${height}px` }}>
          <svg
            width="100%"
            height="100%"
            viewBox={CHART_CONFIG.svgViewBox}
            preserveAspectRatio={CHART_CONFIG.preserveAspectRatio}
            role="img"
            aria-label={title}
          >
            <GridLines />
            <ConfidenceInterval
              upper={paths.upper}
              lower={paths.lower}
              maxX={chartData.points[chartData.points.length - 1]?.x || 100}
            />
            <ForecastPath d={paths.forecast} />
            <ActualPath d={paths.actual} />
            <DataPoints
              points={chartData.points}
              showActual={showActual}
              maxVal={chartData.maxVal}
            />
          </svg>
          <XAxisLabels data={data} />
          <YAxisLabels maxVal={chartData.maxVal} />
        </div>

        <ChartStats data={data} />
      </Card>
    </ErrorBoundary>
  )
}