import { useMemo } from 'react'
import { Card } from './components/Card'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Spacing } from './styles/constants'
import {
  ChartStats,
  ChartLegend,
  YAxisLabels,
  XAxisLabels,
  DataPoints,
  GridLines,
  ConfidenceInterval,
  ForecastPath,
  ActualPath,
  ChartHeader,
  type DemandForecastPoint
} from './chart'

// ============ 常量配置 ============
const CHART_CONFIG = {
  yPaddingTop: 10,
  yScale: 80,
  svgViewBox: '0 0 100 100',
  preserveAspectRatio: 'none' as const
} as const

export interface DemandChartProps {
  data: DemandForecastPoint[]
  title?: string
  showConfidenceInterval?: boolean
  showActual?: boolean
  height?: number
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
