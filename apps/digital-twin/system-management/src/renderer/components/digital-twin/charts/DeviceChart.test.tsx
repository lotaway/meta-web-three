import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { DeviceChart, StatsCard } from '../components/digital-twin/charts/DeviceChart'

describe('DeviceChart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  const mockData = [
    { timestamp: 1000000, value: 10 },
    { timestamp: 1000060, value: 20 },
    { timestamp: 1000120, value: 15 },
    { timestamp: 1000180, value: 25 },
    { timestamp: 1000240, value: 30 },
  ]

  it('should render chart title', () => {
    render(<DeviceChart title="设备效率" data={mockData} />)
    expect(screen.getByText('设备效率')).toBeInTheDocument()
  })

  it('should render with custom height', () => {
    const { container } = render(<DeviceChart title="设备效率" data={mockData} height={300} />)
    const canvas = container.querySelector('canvas')
    expect(canvas).toHaveAttribute('style', expect.stringContaining('300px'))
  })

  it('should render without data', () => {
    const { container } = render(<DeviceChart title="设备效率" data={[]} />)
    const canvas = container.querySelector('canvas')
    expect(canvas).toBeInTheDocument()
  })
})

describe('StatsCard', () => {
  it('should render title and value', () => {
    render(<StatsCard title="平均效率" value={85.5} unit="%" />)
    expect(screen.getByText('平均效率')).toBeInTheDocument()
    expect(screen.getByText('85.5%')).toBeInTheDocument()
  })

  it('should render with number value', () => {
    render(<StatsCard title="设备数量" value={42} />)
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('should render positive change with up arrow', () => {
    render(<StatsCard title="产能" value={100} unit="件" change={5.5} />)
    expect(screen.getByText('↑ 5.5%')).toBeInTheDocument()
  })

  it('should render negative change with down arrow', () => {
    render(<StatsCard title="故障率" value={2.3} unit="%" change={-1.2} />)
    expect(screen.getByText('↓ 1.2%')).toBeInTheDocument()
  })

  it('should not render change when undefined', () => {
    const { container } = render(<StatsCard title="设备数量" value={42} />)
    expect(container.querySelectorAll('div').length).toBe(2) // title + value only
  })
})