import { onBeforeUnmount, onMounted, ref, type Ref } from 'vue'
import { useUserStore } from '@/stores/user'
import type { Alert, StatsSummary } from '@/apis/digitalTwin'
import {
  applyAlertCreated,
  applyProductionOutput,
  DigitalTwinEventType,
  isStaleEvent,
  nextDedupState,
  type DigitalTwinEvent,
  type EventDedupState,
  type AlertCreatedData,
  type ProductionOutputData,
} from '@/utils/digitalTwinEvents'

function buildWsUrl(token: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = import.meta.env.VITE_WS_HOST || window.location.hostname
  const port = import.meta.env.VITE_WS_PORT || '10081'
  return `${protocol}//${host}:${port}/factory-service/ws/digital-twin?token=${encodeURIComponent(token)}`
}

export function useDigitalTwinEvents(
  alerts: Ref<Alert[]>,
  stats: Ref<StatsSummary>,
) {
  const userStore = useUserStore()
  const connected = ref(false)
  let socket: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let dedup: EventDedupState = { lastTimestamp: 0, seenIds: [] }

  function handleMessage(raw: string) {
    let event: DigitalTwinEvent
    try {
      event = JSON.parse(raw) as DigitalTwinEvent
    } catch {
      return
    }
    if (!event.data?.eventId || isStaleEvent(event, dedup)) return
    dedup = nextDedupState(event, dedup)

    if (event.type === DigitalTwinEventType.ALERT_CREATED) {
      alerts.value = applyAlertCreated(event as unknown as DigitalTwinEvent<AlertCreatedData>, alerts.value)
    } else if (event.type === DigitalTwinEventType.PRODUCTION_OUTPUT_UPDATED) {
      stats.value = applyProductionOutput(event as unknown as DigitalTwinEvent<ProductionOutputData>, stats.value)
    }
  }

  function connect() {
    const token = userStore.userInfo.token
    if (!token) return
    socket = new WebSocket(buildWsUrl(token))
    socket.onopen = () => { connected.value = true }
    socket.onmessage = (msg) => handleMessage(String(msg.data))
    socket.onclose = () => {
      connected.value = false
      reconnectTimer = setTimeout(connect, 3000)
    }
    socket.onerror = () => { socket?.close() }
  }

  function disconnect() {
    if (reconnectTimer) clearTimeout(reconnectTimer)
    socket?.close()
    socket = null
    connected.value = false
  }

  onMounted(connect)
  onBeforeUnmount(disconnect)

  return { connected, disconnect }
}
