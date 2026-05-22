import { useCallback, useEffect, useRef, useState } from 'react'
import type { Device } from '../components/digital-twin/scene/FactoryScene'
import type { Alert } from '../components/digital-twin/alert/AlertPanel'
import { DIGITAL_TWIN_WS_URL } from '../config/digital-twin'
import {
  digitalTwinApi,
  parseWsEventData,
  type ChartPoint,
  type DigitalTwinStatsSummary,
} from '../services/digital-twin-api'
import { DigitalTwinWebSocket } from '../services/websocket'

const POLL_INTERVAL_MS = 15000
const MAX_CHART_POINTS = 40

function appendChartPoint(prev: ChartPoint[], value: number): ChartPoint[] {
  const next = [...prev, { timestamp: Date.now(), value }]
  return next.length > MAX_CHART_POINTS ? next.slice(-MAX_CHART_POINTS) : next
}

export function useDigitalTwinData() {
  const [devices, setDevices] = useState<Device[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [stats, setStats] = useState<DigitalTwinStatsSummary | null>(null)
  const [chartData, setChartData] = useState<ChartPoint[]>([])
  const [utilizationChartData, setUtilizationChartData] = useState<ChartPoint[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const wsRef = useRef<DigitalTwinWebSocket | null>(null)

  const refreshAll = useCallback(async () => {
    try {
      const deviceList = await digitalTwinApi.fetchDevices()
      const [alertList, summary] = await Promise.all([
        digitalTwinApi.fetchActiveAlerts(deviceList),
        digitalTwinApi.fetchStatsSummary(),
      ])
      setDevices(deviceList)
      setAlerts(alertList)
      setStats(summary)
      setApiAvailable(true)
      setLoadError(null)
      if (summary.averageEfficiency > 0) {
        setUtilizationChartData((prev) =>
          appendChartPoint(prev, summary.averageEfficiency),
        )
      }
    } catch (err) {
      setApiAvailable(false)
      setLoadError(err instanceof Error ? err.message : '无法连接数字孪生服务')
    }
  }, [])

  const patchDeviceFromEvent = useCallback((payload: Record<string, unknown>) => {
    const deviceCode = payload.deviceCode as string | undefined
    if (!deviceCode) return

    setDevices((prev) =>
      prev.map((d) => {
        if (d.code !== deviceCode) return d
        const updated = { ...d }
        if (payload.status) {
          const status = String(payload.status).toLowerCase()
          if (
            ['online', 'offline', 'running', 'idle', 'warning', 'error'].includes(status)
          ) {
            updated.status = status as Device['status']
          }
        }
        const position = payload.position as { x?: number; y?: number; z?: number } | undefined
        if (position) {
          updated.position = [
            position.x ?? d.position[0],
            position.y ?? d.position[1],
            position.z ?? d.position[2],
          ]
        }
        if (payload.rotation !== undefined) {
          updated.rotation = Number(payload.rotation)
        }
        return updated
      }),
    )
  }, [])

  useEffect(() => {
    void refreshAll()
    const pollId = window.setInterval(() => void refreshAll(), POLL_INTERVAL_MS)
    return () => window.clearInterval(pollId)
  }, [refreshAll])

  useEffect(() => {
    const ws = new DigitalTwinWebSocket({
      url: DIGITAL_TWIN_WS_URL,
      onOpen: () => setIsConnected(true),
      onClose: () => setIsConnected(false),
      reconnectAttempts: 10,
    })

    const handleMessage = (message: { type: string; data: unknown }) => {
      const payload = parseWsEventData(message.data)

      switch (message.type) {
        case 'DEVICE_STATUS_CHANGED':
        case 'AGV_POSITION_UPDATED':
          patchDeviceFromEvent(payload)
          break
        case 'DEVICE_POSITION_UPDATED':
          patchDeviceFromEvent(payload)
          break
        case 'ALERT_CREATED':
          void refreshAll()
          break
        case 'PRODUCTION_OUTPUT_UPDATED': {
          const output = Number(payload.output)
          if (!Number.isNaN(output)) {
            setChartData((prev) => appendChartPoint(prev, output))
          }
          void refreshAll()
          break
        }
        default:
          break
      }
    }

    ws.subscribe('*', handleMessage)
    ws.connect()
    wsRef.current = ws

    return () => {
      ws.disconnect()
      wsRef.current = null
    }
  }, [patchDeviceFromEvent, refreshAll])

  const acknowledgeAlert = useCallback(
    async (alertId: string) => {
      await digitalTwinApi.acknowledgeAlert(alertId)
      await refreshAll()
    },
    [refreshAll],
  )

  const resolveAlert = useCallback(
    async (alertId: string) => {
      await digitalTwinApi.resolveAlert(alertId)
      await refreshAll()
    },
    [refreshAll],
  )

  return {
    devices,
    alerts,
    stats,
    chartData,
    utilizationChartData,
    isConnected,
    apiAvailable,
    loadError,
    refreshAll,
    acknowledgeAlert,
    resolveAlert,
  }
}
