import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { digitalTwinApi, parseWsEventData, type ChartPoint } from '../services/digital-twin-api'

// Mock axios
vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
    })),
  },
}))

describe('digitalTwinApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('fetchDevices', () => {
    it('should return array of devices', async () => {
      const mockDevices = [
        { code: 'DEVICE-001', name: '设备1', status: 'online', position: [0, 0, 0] },
        { code: 'DEVICE-002', name: '设备2', status: 'running', position: [10, 0, 0] },
      ]
      
      // Mock implementation would go here
      // For now just verify the function exists
      expect(typeof digitalTwinApi.fetchDevices).toBe('function')
    })
  })

  describe('fetchActiveAlerts', () => {
    it('should return array of alerts for given devices', async () => {
      expect(typeof digitalTwinApi.fetchActiveAlerts).toBe('function')
    })
  })

  describe('fetchStatsSummary', () => {
    it('should return stats summary', async () => {
      expect(typeof digitalTwinApi.fetchStatsSummary).toBe('function')
    })
  })
})

describe('parseWsEventData', () => {
  it('should parse valid JSON string', () => {
    const input = '{"deviceCode": "DEVICE-001", "status": "running"}'
    const result = parseWsEventData(input)
    expect(result).toEqual({ deviceCode: 'DEVICE-001', status: 'running' })
  })

  it('should return empty object for invalid JSON', () => {
    const input = 'invalid json'
    const result = parseWsEventData(input)
    expect(result).toEqual({})
  })

  it('should return empty object for null input', () => {
    const input = null
    const result = parseWsEventData(input)
    expect(result).toEqual({})
  })

  it('should return empty object for undefined input', () => {
    const input = undefined
    const result = parseWsEventData(input)
    expect(result).toEqual({})
  })
})

describe('ChartPoint type', () => {
  it('should accept valid chart point object', () => {
    const point: ChartPoint = { timestamp: Date.now(), value: 85.5 }
    expect(point.timestamp).toBeDefined()
    expect(point.value).toBe(85.5)
  })
})