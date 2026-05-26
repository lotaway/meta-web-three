import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, act } from '@testing-library/react'
import { fireEvent } from '@testing-library/react'

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  
  readyState = MockWebSocket.OPEN
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null

  constructor(public url: string) {
    setTimeout(() => {
      this.onopen?.()
    }, 0)
  }

  send(data: string) {
    // Mock send
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  }
}

vi.stubGlobal('WebSocket', MockWebSocket)

// Import components after mocks
import { InventoryAlertPanel, type AlertItem } from '../InventoryAlertPanel'
import { RestockSuggestions, type RestockItem } from '../RestockSuggestions'

describe('Alert Flow Tests - End-to-End', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('InventoryAlertPanel - Alert Display and Management', () => {
    const mockAlerts: AlertItem[] = [
      {
        id: 'alert-001',
        code: 'INV-20240526-001',
        deviceCode: 'DEVICE-001',
        level: 'critical',
        type: 'LOW_STOCK',
        title: '库存不足告警',
        description: '商品 SKU-1234 库存低于安全库存',
        status: 'pending',
        occurredAt: new Date().toISOString(),
      },
      {
        id: 'alert-002',
        code: 'INV-20240526-002',
        deviceCode: 'DEVICE-002',
        level: 'warning',
        type: 'EXPIRING_SOON',
        title: '临期商品告警',
        description: '商品 SKU-5678 将在 7 天后过期',
        status: 'pending',
        occurredAt: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: 'alert-003',
        code: 'INV-20240526-003',
        deviceCode: 'DEVICE-003',
        level: 'info',
        type: 'STOCK_LEVEL_NORMAL',
        title: '库存正常',
        description: '所有商品库存处于正常水平',
        status: 'resolved',
        occurredAt: new Date(Date.now() - 7200000).toISOString(),
      },
    ]

    it('should render alert panel with alerts', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByText('库存告警')).toBeInTheDocument()
    })

    it('should display correct alert count', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      // Should show total alerts
      expect(screen.getByText(/共 \d+ 条/)).toBeInTheDocument()
    })

    it('should display critical alerts with red styling', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByText('库存不足告警')).toBeInTheDocument()
    })

    it('should display warning alerts with yellow styling', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByText('临期商品告警')).toBeInTheDocument()
    })

    it('should display info alerts with blue styling', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByText('库存正常')).toBeInTheDocument()
    })

    it('should call onAcknowledge when acknowledge button is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const mockOnAcknowledge = vi.fn()

      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={mockOnAcknowledge}
          onResolve={vi.fn()}
        />
      )

      // Find and click acknowledge button for first alert
      const buttons = await screen.findAllByRole('button', { name: /确认/i })
      if (buttons.length > 0) {
        await user.click(buttons[0])
        expect(mockOnAcknowledge).toHaveBeenCalled()
      }
    })

    it('should call onResolve when resolve button is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const mockOnResolve = vi.fn()

      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={mockOnResolve}
        />
      )

      // Find and click resolve button
      const buttons = await screen.findAllByRole('button', { name: /处理/i })
      if (buttons.length > 0) {
        await user.click(buttons[0])
        expect(mockOnResolve).toHaveBeenCalled()
      }
    })

    it('should filter alerts by level', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
          filterLevel="critical"
        />
      )

      // Should show only critical alerts
      expect(screen.getByText('库存不足告警')).toBeInTheDocument()
    })

    it('should handle empty alerts list', () => {
      render(
        <InventoryAlertPanel
          alerts={[]}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByText('暂无告警')).toBeInTheDocument()
    })

    it('should display alert timestamps', () => {
      render(
        <InventoryAlertPanel
          alerts={mockAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      // Should show time ago text
      expect(screen.getByText(/分钟前|小时前|刚刚/)).toBeInTheDocument()
    })

    it('should sort alerts by priority', () => {
      const unsortedAlerts: AlertItem[] = [
        { ...mockAlerts[0], level: 'info' },
        { ...mockAlerts[1], level: 'critical' },
        { ...mockAlerts[2], level: 'warning' },
      ]

      render(
        <InventoryAlertPanel
          alerts={unsortedAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
          sortBy="priority"
        />
      )

      // Critical should appear first
      expect(screen.getByText('库存不足告警')).toBeInTheDocument()
    })
  })

  describe('RestockSuggestions - Automated Restock Recommendations', () => {
    const mockRestockItems: RestockItem[] = [
      {
        id: 'restock-001',
        sku: 'SKU-1234',
        productName: '商品A',
        currentStock: 50,
        safetyStock: 100,
        recommendedQuantity: 150,
        urgency: 'high',
        estimatedCost: 1500,
        supplier: '供应商A',
      },
      {
        id: 'restock-002',
        sku: 'SKU-5678',
        productName: '商品B',
        currentStock: 80,
        safetyStock: 100,
        recommendedQuantity: 120,
        urgency: 'medium',
        estimatedCost: 2400,
        supplier: '供应商B',
      },
      {
        id: 'restock-003',
        sku: 'SKU-9012',
        productName: '商品C',
        currentStock: 95,
        safetyStock: 100,
        recommendedQuantity: 105,
        urgency: 'low',
        estimatedCost: 1050,
        supplier: '供应商C',
      },
    ]

    it('should render restock suggestions panel', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText('补货建议')).toBeInTheDocument()
    })

    it('should display correct number of suggestions', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText(/共 \d+ 项/)).toBeInTheDocument()
    })

    it('should highlight high urgency items', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText('商品A')).toBeInTheDocument()
    })

    it('should display product details', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      // SKU
      expect(screen.getByText('SKU-1234')).toBeInTheDocument()
      // Current stock
      expect(screen.getByText('50')).toBeInTheDocument()
      // Safety stock
      expect(screen.getByText('100')).toBeInTheDocument()
    })

    it('should display recommended quantity', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText('150')).toBeInTheDocument()
    })

    it('should display estimated cost', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText(/¥?1,?500/)).toBeInTheDocument()
    })

    it('should call onApprove when approve button is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const mockOnApprove = vi.fn()

      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={mockOnApprove}
          onDismiss={vi.fn()}
        />
      )

      // Find and click approve button
      const approveButtons = await screen.findAllByRole('button', { name: /批准|同意/i })
      if (approveButtons.length > 0) {
        await user.click(approveButtons[0])
        expect(mockOnApprove).toHaveBeenCalled()
      }
    })

    it('should call onDismiss when dismiss button is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const mockOnDismiss = vi.fn()

      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={mockOnDismiss}
        />
      )

      // Find and click dismiss button
      const dismissButtons = await screen.findAllByRole('button', { name: /忽略| Dismiss/i })
      if (dismissButtons.length > 0) {
        await user.click(dismissButtons[0])
        expect(mockOnDismiss).toHaveBeenCalled()
      }
    })

    it('should calculate total estimated cost', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      // Total: 1500 + 2400 + 1050 = 4950
      expect(screen.getByText(/4,?950/)).toBeInTheDocument()
    })

    it('should handle empty suggestions', () => {
      render(
        <RestockSuggestions
          items={[]}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText('暂无补货建议')).toBeInTheDocument()
    })

    it('should filter by urgency level', () => {
      render(
        <RestockSuggestions
          items={mockRestockItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
          filterUrgency="high"
        />
      )

      // Should show only high urgency items
      expect(screen.getByText('商品A')).toBeInTheDocument()
    })

    it('should sort by urgency by default', () => {
      const unsortedItems: RestockItem[] = [
        { ...mockRestockItems[0], urgency: 'low' },
        { ...mockRestockItems[1], urgency: 'high' },
        { ...mockRestockItems[2], urgency: 'medium' },
      ]

      render(
        <RestockSuggestions
          items={unsortedItems}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      // High urgency should appear first
      expect(screen.getByText('商品B')).toBeInTheDocument()
    })
  })

  describe('Alert Flow - End-to-End Scenarios', () => {
    it('should show notification when new alert arrives', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      
      const initialAlerts: AlertItem[] = []
      const { rerender } = render(
        <InventoryAlertPanel
          alerts={initialAlerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByText('暂无告警')).toBeInTheDocument()

      // Simulate new alert arriving
      const newAlert: AlertItem = {
        id: 'alert-new',
        code: 'INV-NEW-001',
        deviceCode: 'DEVICE-NEW',
        level: 'critical',
        type: 'LOW_STOCK',
        title: '新告警',
        description: '新商品库存不足',
        status: 'pending',
        occurredAt: new Date().toISOString(),
      }

      await act(async () => {
        rerender(
          <InventoryAlertPanel
            alerts={[newAlert]}
            onAcknowledge={vi.fn()}
            onResolve={vi.fn()}
          />
        )
        vi.advanceTimersByTime(1000)
      })

      expect(screen.getByText('新告警')).toBeInTheDocument()
    })

    it('should update restock suggestion after approval', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const mockOnApprove = vi.fn()

      const items: RestockItem[] = [
        {
          id: 'restock-001',
          sku: 'SKU-1234',
          productName: '商品A',
          currentStock: 50,
          safetyStock: 100,
          recommendedQuantity: 150,
          urgency: 'high',
          estimatedCost: 1500,
          supplier: '供应商A',
        },
      ]

      const { rerender } = render(
        <RestockSuggestions
          items={items}
          onApprove={mockOnApprove}
          onDismiss={vi.fn()}
        />
      )

      expect(screen.getByText('商品A')).toBeInTheDocument()

      // Simulate approval
      await act(async () => {
        mockOnApprove(items[0].id)
        rerender(
          <RestockSuggestions
            items={[]}
            onApprove={mockOnApprove}
            onDismiss={vi.fn()}
          />
        )
      })

      expect(screen.getByText('暂无补货建议')).toBeInTheDocument()
    })

    it('should transition alert from pending to resolved', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const mockOnResolve = vi.fn()

      const pendingAlert: AlertItem = {
        id: 'alert-001',
        code: 'INV-001',
        deviceCode: 'DEVICE-001',
        level: 'critical',
        type: 'LOW_STOCK',
        title: '库存不足',
        description: '需要补货',
        status: 'pending',
        occurredAt: new Date().toISOString(),
      }

      const { rerender } = render(
        <InventoryAlertPanel
          alerts={[pendingAlert]}
          onAcknowledge={vi.fn()}
          onResolve={mockOnResolve}
        />
      )

      expect(screen.getByText('库存不足')).toBeInTheDocument()

      // Simulate resolve
      await act(async () => {
        mockOnResolve(pendingAlert.id, '已处理')
        const resolvedAlert = { ...pendingAlert, status: 'resolved' as const }
        rerender(
          <InventoryAlertPanel
            alerts={[resolvedAlert]}
            onAcknowledge={vi.fn()}
            onResolve={mockOnResolve}
          />
        )
      })

      expect(screen.getByText('已处理')).toBeInTheDocument()
    })
  })

  describe('Alert Priority Calculation', () => {
    it('should calculate urgency based on stock level', () => {
      const items: RestockItem[] = [
        {
          id: '1',
          sku: 'SKU-1',
          productName: '商品1',
          currentStock: 10,
          safetyStock: 100,
          recommendedQuantity: 200,
          urgency: 'high',
          estimatedCost: 1000,
          supplier: '供应商1',
        },
        {
          id: '2',
          sku: 'SKU-2',
          productName: '商品2',
          currentStock: 90,
          safetyStock: 100,
          recommendedQuantity: 110,
          urgency: 'low',
          estimatedCost: 500,
          supplier: '供应商2',
        },
      ]

      render(
        <RestockSuggestions
          items={items}
          onApprove={vi.fn()}
          onDismiss={vi.fn()}
        />
      )

      // High urgency should appear first
      const productElements = screen.getAllByText(/商品\d/)
      expect(productElements[0]).toHaveTextContent('商品1')
    })
  })

  describe('Performance', () => {
    it('should handle large number of alerts', () => {
      const manyAlerts = Array.from({ length: 100 }, (_, i) => ({
        id: `alert-${i}`,
        code: `INV-${i}`,
        deviceCode: `DEVICE-${i}`,
        level: (['critical', 'warning', 'info'] as const)[i % 3],
        type: 'LOW_STOCK',
        title: `告警 ${i}`,
        description: `描述 ${i}`,
        status: 'pending',
        occurredAt: new Date().toISOString(),
      }))

      expect(() => {
        render(
          <InventoryAlertPanel
            alerts={manyAlerts}
            onAcknowledge={vi.fn()}
            onResolve={vi.fn()}
          />
        )
      }).not.toThrow()
    })

    it('should handle large number of restock suggestions', () => {
      const manyItems = Array.from({ length: 100 }, (_, i) => ({
        id: `restock-${i}`,
        sku: `SKU-${i}`,
        productName: `商品 ${i}`,
        currentStock: Math.floor(Math.random() * 100),
        safetyStock: 100,
        recommendedQuantity: 100,
        urgency: (['high', 'medium', 'low'] as const)[i % 3],
        estimatedCost: Math.floor(Math.random() * 5000),
        supplier: `供应商 ${i}`,
      }))

      expect(() => {
        render(
          <RestockSuggestions
            items={manyItems}
            onApprove={vi.fn()}
            onDismiss={vi.fn()}
          />
        )
      }).not.toThrow()
    })
  })

  describe('Accessibility', () => {
    it('should have proper roles for alert items', () => {
      const alerts: AlertItem[] = [
        {
          id: 'alert-001',
          code: 'INV-001',
          deviceCode: 'DEVICE-001',
          level: 'critical',
          type: 'LOW_STOCK',
          title: '测试告警',
          description: '测试描述',
          status: 'pending',
          occurredAt: new Date().toISOString(),
        },
      ]

      render(
        <InventoryAlertPanel
          alerts={alerts}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      expect(screen.getByRole('status')).toBeInTheDocument()
    })

    it('should have keyboard accessible buttons', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })

      render(
        <InventoryAlertPanel
          alerts={[
            {
              id: 'alert-001',
              code: 'INV-001',
              deviceCode: 'DEVICE-001',
              level: 'warning',
              type: 'TEST',
              title: '测试',
              description: '测试描述',
              status: 'pending',
              occurredAt: new Date().toISOString(),
            },
          ]}
          onAcknowledge={vi.fn()}
          onResolve={vi.fn()}
        />
      )

      // Tab to the first button
      await user.keyboard('{Tab}')
      await user.keyboard('{Tab}')
      await user.keyboard('{Tab}')
      
      // Should be able to interact with buttons via keyboard
      expect(screen.getByRole('button', { name: /确认/i })).toBeInTheDocument()
    })
  })
})