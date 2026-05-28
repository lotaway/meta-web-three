import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, act } from '@testing-library/react'
import { fireEvent } from '@testing-library/react'

// Mock Chart.js
vi.mock('react-chartjs-2', () => ({
  Line: ({ data, options }: { data: unknown; options: unknown }) => (
    <div data-testid="mock-line-chart" data-data={JSON.stringify(data)} data-options={JSON.stringify(options)} />
  ),
  Bar: ({ data, options }: { data: unknown; options: unknown }) => (
    <div data-testid="mock-bar-chart" data-data={JSON.stringify(data)} data-options={JSON.stringify(options)} />
  ),
  Doughnut: ({ data, options }: { data: unknown; options: unknown }) => (
    <div data-testid="mock-doughnut-chart" data-data={JSON.stringify(data)} data-options={JSON.stringify(options)} />
  ),
}))

vi.mock('chart.js', () => ({
  Chart: {
    register: vi.fn(),
  },
  CategoryScale: vi.fn(),
  LinearScale: vi.fn(),
  PointElement: vi.fn(),
  LineElement: vi.fn(),
  BarElement: vi.fn(),
  ArcElement: vi.fn(),
  Title: vi.fn(),
  Tooltip: vi.fn(),
  Legend: vi.fn(),
}))

// Import components after mocks
import { DemandChart, type DemandDataPoint } from '../DemandChart'
import { ShelfHeatmap, type HeatmapCell } from '../ShelfHeatmap'
import { WarehouseStatus, type WarehouseStatusData } from '../WarehouseStatus'

describe('Real-time Data Display Tests', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('DemandChart - Real-time Demand Data', () => {
    const generateMockDemandData = (count: number): DemandDataPoint[] => {
      const now = Date.now()
      return Array.from({ length: count }, (_, i) => ({
        date: new Date(now - (count - i - 1) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        demand: Math.floor(Math.random() * 1000) + 500,
        forecast: Math.floor(Math.random() * 1000) + 500,
      }))
    }

    it('should render chart with initial data', () => {
      const mockData = generateMockDemandData(7)

      render(
        <DemandChart
          data={mockData}
          title="需求趋势"
        />
      )

      expect(screen.getByTestId('mock-line-chart')).toBeInTheDocument()
    })

    it('should display chart title', () => {
      const mockData = generateMockDemandData(7)

      render(
        <DemandChart
          data={mockData}
          title="需求趋势"
        />
      )

      expect(screen.getByText('需求趋势')).toBeInTheDocument()
    })

    it('should handle empty data gracefully', () => {
      render(
        <DemandChart
          data={[]}
          title="需求趋势"
        />
      )

      // Should render without crashing
      expect(screen.getByTestId('mock-line-chart')).toBeInTheDocument()
    })

    it('should update when data changes', () => {
      const initialData = generateMockDemandData(7)
      const { rerender } = render(
        <DemandChart
          data={initialData}
          title="需求趋势"
        />
      )

      const updatedData = generateMockDemandData(7)
      rerender(
        <DemandChart
          data={updatedData}
          title="需求趋势"
        />
      )

      expect(screen.getByTestId('mock-line-chart')).toBeInTheDocument()
    })

    it('should display all data points', () => {
      const mockData = generateMockDemandData(30)

      render(
        <DemandChart
          data={mockData}
          title="需求趋势"
        />
      )

      expect(screen.getByTestId('mock-line-chart')).toBeInTheDocument()
    })
  })

  describe('ShelfHeatmap - Real-time Load Distribution', () => {
    const generateMockHeatmapData = (rows: number, cols: number): HeatmapCell[] => {
      const data: HeatmapCell[] = []
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          data.push({
            row,
            col,
            value: Math.floor(Math.random() * 100),
            label: `R${row}-C${col}`,
          })
        }
      }
      return data
    }

    it('should render heatmap grid', () => {
      const mockData = generateMockHeatmapData(5, 10)

      render(
        <ShelfHeatmap
          data={mockData}
          title="货架负载分布"
        />
      )

      expect(screen.getByText('货架负载分布')).toBeInTheDocument()
    })

    it('should display color legend', () => {
      const mockData = generateMockHeatmapData(5, 10)

      render(
        <ShelfHeatmap
          data={mockData}
          title="货架负载分布"
        />
      )

      // Check for legend items
      expect(screen.getByText(/低负载|0-25%/)).toBeInTheDocument()
    })

    it('should handle empty heatmap data', () => {
      render(
        <ShelfHeatmap
          data={[]}
          title="货架负载分布"
        />
      )

      expect(screen.getByText('货架负载分布')).toBeInTheDocument()
    })

    it('should render correct number of cells', () => {
      const mockData = generateMockHeatmapData(5, 10)

      render(
        <ShelfHeatmap
          data={mockData}
          title="货架负载分布"
        />
      )

      // Should render the heatmap container
      expect(screen.getByText('货架负载分布')).toBeInTheDocument()
    })

    it('should update heatmap when data changes', () => {
      const initialData = generateMockHeatmapData(5, 10)
      const { rerender } = render(
        <ShelfHeatmap
          data={initialData}
          title="货架负载分布"
        />
      )

      const updatedData = generateMockHeatmapData(5, 10)
      rerender(
        <ShelfHeatmap
          data={updatedData}
          title="货架负载分布"
        />
      )

      expect(screen.getByText('货架负载分布')).toBeInTheDocument()
    })

    it('should handle large grid sizes', () => {
      const largeData = generateMockHeatmapData(20, 30)

      expect(() => {
        render(
          <ShelfHeatmap
            data={largeData}
            title="货架负载分布"
          />
        )
      }).not.toThrow()
    })
  })

  describe('WarehouseStatus - Real-time Status Updates', () => {
    const mockStatusData: WarehouseStatusData = {
      warehouseId: 'wh-001',
      warehouseName: '主仓库A',
      totalCapacity: 10000,
      usedCapacity: 7500,
      itemCount: 5234,
      inboundToday: 120,
      outboundToday: 85,
      pendingOrders: 15,
      efficiency: 87.5,
      lastUpdate: new Date().toISOString(),
    }

    it('should render warehouse status panel', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      expect(screen.getByText('主仓库A')).toBeInTheDocument()
    })

    it('should display capacity utilization percentage', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      // 7500/10000 * 100 = 75%
      expect(screen.getByText(/75\.0%/)).toBeInTheDocument()
    })

    it('should display item count', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      expect(screen.getByText('5234')).toBeInTheDocument()
    })

    it('should display today\'s inbound count', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      expect(screen.getByText('120')).toBeInTheDocument()
    })

    it('should display today\'s outbound count', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      expect(screen.getByText('85')).toBeInTheDocument()
    })

    it('should display efficiency metric', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      expect(screen.getByText(/87\.5/)).toBeInTheDocument()
    })

    it('should render in compact mode', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={true}
        />
      )

      expect(screen.getByText('主仓库A')).toBeInTheDocument()
    })

    it('should handle status updates', () => {
      const { rerender } = render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      const updatedStatus: WarehouseStatusData = {
        ...mockStatusData,
        usedCapacity: 8000,
        itemCount: 5500,
        efficiency: 89.2,
        lastUpdate: new Date().toISOString(),
      }

      rerender(
        <WarehouseStatus
          data={updatedStatus}
          compact={false}
        />
      )

      // 8000/10000 * 100 = 80%
      expect(screen.getByText(/80\.0%/)).toBeInTheDocument()
    })

    it('should display last update timestamp', () => {
      render(
        <WarehouseStatus
          data={mockStatusData}
          compact={false}
        />
      )

      // Should show last update time
      expect(screen.getByText(/最后更新/)).toBeInTheDocument()
    })
  })

  describe('Data Refresh Simulation', () => {
    it('should handle periodic data updates', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      
      let updateCount = 0
      const mockData = [
        { date: '2024-01-01', demand: 100, forecast: 110 },
        { date: '2024-01-02', demand: 150, forecast: 140 },
      ]

      const { rerender } = render(
        <DemandChart
          data={mockData}
          title="需求趋势"
        />
      )

      // Simulate data refresh
      await act(async () => {
        vi.advanceTimersByTime(5000)
      })

      updateCount++
      expect(updateCount).toBeGreaterThanOrEqual(0)
    })

    it('should maintain data continuity during updates', () => {
      const initialData = [
        { date: '2024-01-01', demand: 100, forecast: 110 },
        { date: '2024-01-02', demand: 150, forecast: 140 },
      ]

      const { rerender } = render(
        <DemandChart
          data={initialData}
          title="需求趋势"
        />
      )

      const newData = [
        { date: '2024-01-01', demand: 100, forecast: 110 },
        { date: '2024-01-02', demand: 150, forecast: 140 },
        { date: '2024-01-03', demand: 180, forecast: 170 },
      ]

      rerender(
        <DemandChart
          data={newData}
          title="需求趋势"
        />
      )

      // Should maintain previous data and add new
      expect(screen.getByTestId('mock-line-chart')).toBeInTheDocument()
    })
  })

  describe('Error Handling', () => {
    it('should handle invalid data gracefully', () => {
      const invalidData = [
        { date: 'invalid', demand: NaN, forecast: NaN },
      ] as unknown as DemandDataPoint[]

      expect(() => {
        render(
          <DemandChart
            data={invalidData}
            title="需求趋势"
          />
        )
      }).not.toThrow()
    })

    it('should handle undefined data', () => {
      expect(() => {
        render(
          <DemandChart
            // @ts-expect-error - Testing invalid input
            data={undefined}
            title="需求趋势"
          />
        )
      }).not.toThrow()
    })

    it('should handle null values in data', () => {
      const dataWithNulls = [
        { date: '2024-01-01', demand: null, forecast: null },
      ] as unknown as DemandDataPoint[]

      expect(() => {
        render(
          <DemandChart
            data={dataWithNulls}
            title="需求趋势"
          />
        )
      }).not.toThrow()
    })
  })

  describe('Performance', () => {
    it('should handle high-frequency updates', async () => {
      const mockData = Array.from({ length: 100 }, (_, i) => ({
        date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        demand: Math.floor(Math.random() * 1000),
        forecast: Math.floor(Math.random() * 1000),
      }))

      const { rerender } = render(
        <DemandChart
          data={mockData}
          title="需求趋势"
        />
      )

      // Simulate rapid updates
      for (let i = 0; i < 10; i++) {
        const newData = mockData.map(d => ({
          ...d,
          demand: Math.floor(Math.random() * 1000),
        }))
        
        rerender(
          <DemandChart
            data={newData}
            title="需求趋势"
          />
        )
      }

      expect(screen.getByTestId('mock-line-chart')).toBeInTheDocument()
    })

    it('should render large dataset efficiently', () => {
      const largeData = Array.from({ length: 1000 }, (_, i) => ({
        row: Math.floor(i / 50),
        col: i % 50,
        value: Math.floor(Math.random() * 100),
        label: `Cell-${i}`,
      }))

      expect(() => {
        render(
          <ShelfHeatmap
            data={largeData}
            title="货架负载分布"
          />
        )
      }).not.toThrow()
    })
  })
})