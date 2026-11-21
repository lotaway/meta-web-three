import type { Alert, StatsSummary } from '@/apis/digitalTwin'

export const DigitalTwinEventType = {
  DEVICE_STATUS_CHANGED: 'DEVICE_STATUS_CHANGED',
  DEVICE_POSITION_UPDATED: 'DEVICE_POSITION_UPDATED',
  PRODUCTION_OUTPUT_UPDATED: 'PRODUCTION_OUTPUT_UPDATED',
  ALERT_CREATED: 'ALERT_CREATED',
  AGV_POSITION_UPDATED: 'AGV_POSITION_UPDATED',
} as const

export type DigitalTwinEventTypeName =
  (typeof DigitalTwinEventType)[keyof typeof DigitalTwinEventType]

export interface DigitalTwinEvent<T = Record<string, unknown>> {
  type: DigitalTwinEventTypeName
  data: T & {
    eventId: string
    eventType: string
    timestamp: number
  }
}

export interface AlertCreatedData {
  alertId: number
  alertCode: string
  deviceCode: string
  workshopId: string
  level: string
  type: string
  title: string
  description: string
  status: string
  occurredAt: string
  createdAt: string
}

export interface ProductionOutputData {
  lineCode: string
  output: number
  stats: StatsSummary
}

export interface EventDedupState {
  lastTimestamp: number
  seenIds: string[]
}

export function isStaleEvent(event: DigitalTwinEvent, state: EventDedupState): boolean {
  if (event.data.timestamp < state.lastTimestamp) return true
  return state.seenIds.includes(event.data.eventId)
}

export function applyAlertCreated(event: DigitalTwinEvent<AlertCreatedData>, alerts: Alert[]): Alert[] {
  const data = event.data
  const alert: Alert = {
    id: data.alertId,
    alertCode: data.alertCode,
    deviceCode: data.deviceCode,
    workshopId: data.workshopId,
    level: data.level,
    type: data.type,
    title: data.title,
    description: data.description,
    status: data.status,
    solution: '',
    acknowledgedBy: '',
    resolvedBy: '',
    occurredAt: data.occurredAt,
    acknowledgedAt: '',
    resolvedAt: '',
    createdAt: data.createdAt,
  }
  if (alerts.some((item) => item.alertCode === alert.alertCode)) return alerts
  return [alert, ...alerts]
}

export function applyProductionOutput(event: DigitalTwinEvent<ProductionOutputData>, stats: StatsSummary): StatsSummary {
  return event.data.stats ?? stats
}

export function nextDedupState(event: DigitalTwinEvent, state: EventDedupState): EventDedupState {
  return {
    lastTimestamp: Math.max(state.lastTimestamp, event.data.timestamp),
    seenIds: [event.data.eventId, ...state.seenIds].slice(0, 200),
  }
}
