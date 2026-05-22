import axios from 'axios'
import type { Device } from '../components/digital-twin/scene/FactoryScene'
import type { Alert } from '../components/digital-twin/alert/AlertPanel'
import { DIGITAL_TWIN_API_BASE_URL } from '../config/digital-twin'

export interface DigitalTwinStatsSummary {
  onlineDeviceCount: number
  activeAlertCount: number
  averageEfficiency: number
}

export interface ChartPoint {
  timestamp: number
  value: number
}

interface ApiDevice {
  id: number
  deviceCode: string
  deviceName: string
  deviceType: string
  status: string
  positionX?: number | null
  positionY?: number | null
  positionZ?: number | null
  rotationY?: number | null
}

interface ApiAlert {
  id: number
  alertCode: string
  deviceCode: string
  level: string
  type: string
  title: string
  description: string
  status: string
  occurredAt?: string
}

const client = axios.create({
  baseURL: DIGITAL_TWIN_API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
})

const statusMap: Record<string, Device['status']> = {
  ONLINE: 'online',
  OFFLINE: 'offline',
  RUNNING: 'running',
  IDLE: 'idle',
  WARNING: 'warning',
  ERROR: 'error',
  MAINTENANCE: 'idle',
}

const alertLevelMap: Record<string, Alert['level']> = {
  INFO: 'info',
  WARNING: 'warning',
  ERROR: 'error',
  CRITICAL: 'critical',
}

const alertStatusMap: Record<string, Alert['status']> = {
  TRIGGERED: 'triggered',
  ACKNOWLEDGED: 'acknowledged',
  IN_PROGRESS: 'in_progress',
  RESOLVED: 'resolved',
  CLOSED: 'resolved',
}

export function mapApiDevice(d: ApiDevice, deviceNameByCode?: Map<string, string>): Device {
  const x = d.positionX ?? 0
  const y = d.positionY ?? 0.25
  const z = d.positionZ ?? 0
  return {
    id: String(d.id),
    code: d.deviceCode,
    name: d.deviceName,
    type: d.deviceType,
    status: statusMap[d.status] ?? 'offline',
    position: [x, y, z],
    rotation: d.rotationY ?? 0,
  }
}

export function mapApiAlert(a: ApiAlert, deviceNameByCode: Map<string, string>): Alert {
  return {
    id: String(a.id),
    code: a.alertCode,
    deviceCode: a.deviceCode,
    deviceName: deviceNameByCode.get(a.deviceCode) ?? a.deviceCode,
    level: alertLevelMap[a.level] ?? 'info',
    type: a.type,
    title: a.title,
    description: a.description,
    status: alertStatusMap[a.status] ?? 'triggered',
    occurredAt: a.occurredAt ?? new Date().toISOString(),
  }
}

export function parseWsEventData(data: unknown): Record<string, unknown> {
  if (typeof data === 'string') {
    try {
      return JSON.parse(data) as Record<string, unknown>
    } catch {
      return {}
    }
  }
  if (data && typeof data === 'object') {
    return data as Record<string, unknown>
  }
  return {}
}

export const digitalTwinApi = {
  async fetchDevices(): Promise<Device[]> {
    const { data } = await client.get<ApiDevice[]>('/api/digital-twin/devices')
    return (data ?? []).map((d) => mapApiDevice(d))
  },

  async fetchActiveAlerts(devices: Device[]): Promise<Alert[]> {
    const nameByCode = new Map(devices.map((d) => [d.code, d.name]))
    const { data } = await client.get<ApiAlert[]>('/api/digital-twin/alerts/active')
    return (data ?? []).map((a) => mapApiAlert(a, nameByCode))
  },

  async fetchStatsSummary(): Promise<DigitalTwinStatsSummary> {
    const { data } = await client.get<DigitalTwinStatsSummary>('/api/digital-twin/stats/summary')
    return data
  },

  async acknowledgeAlert(alertId: string, acknowledgedBy = 'operator'): Promise<void> {
    await client.post(`/api/digital-twin/alert/${alertId}/acknowledge`, { acknowledgedBy })
  },

  async resolveAlert(alertId: string, resolvedBy = 'operator', solution = ''): Promise<void> {
    await client.post(`/api/digital-twin/alert/${alertId}/resolve`, { resolvedBy, solution })
  },

  isReachable(): Promise<boolean> {
    return client
      .get('/api/digital-twin/stats/summary')
      .then(() => true)
      .catch(() => false)
  },
}
