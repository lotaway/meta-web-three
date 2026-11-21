import { describe, expect, it } from 'vitest'
import {
  applyAlertCreated,
  applyProductionOutput,
  isStaleEvent,
  nextDedupState,
  type DigitalTwinEvent,
  type AlertCreatedData,
  type EventDedupState,
} from '@/utils/digitalTwinEvents'

const alertEvent = (timestamp: number, eventId: string): DigitalTwinEvent<AlertCreatedData> => ({
  type: 'ALERT_CREATED',
  data: {
    eventId,
    eventType: 'ALERT_CREATED',
    timestamp,
    alertId: 1,
    alertCode: 'ALERT-1',
    deviceCode: 'DEV-1',
    workshopId: 'WS-1',
    level: 'WARNING',
    type: 'DEVICE_ERROR',
    title: 'High temperature',
    description: 'Temperature exceeded threshold',
    status: 'TRIGGERED',
    occurredAt: '2026-11-21T10:00:00',
    createdAt: '2026-11-21T10:00:00',
  },
})

describe('digital twin event payloads', () => {
  it('builds a complete alert directly from the event payload', () => {
    const result = applyAlertCreated(alertEvent(1000, 'e1'), [])
    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({
      id: 1,
      alertCode: 'ALERT-1',
      title: 'High temperature',
      status: 'TRIGGERED',
      occurredAt: '2026-11-21T10:00:00',
    })
  })

  it('does not duplicate an alert already in the list', () => {
    const existing = applyAlertCreated(alertEvent(1000, 'e1'), [])
    const result = applyAlertCreated(alertEvent(2000, 'e2'), existing)
    expect(result).toHaveLength(1)
  })

  it('applies production stats without a refetch', () => {
    const event: DigitalTwinEvent = {
      type: 'PRODUCTION_OUTPUT_UPDATED',
      data: {
        eventId: 'e3',
        eventType: 'PRODUCTION_OUTPUT_UPDATED',
        timestamp: 1000,
        lineCode: 'LINE-1',
        output: 42,
        stats: { onlineDeviceCount: 5, activeAlertCount: 2, averageEfficiency: 0.9 },
      },
    }
    expect(applyProductionOutput(event as any, { onlineDeviceCount: 0, activeAlertCount: 0, averageEfficiency: 0 }))
      .toEqual({ onlineDeviceCount: 5, activeAlertCount: 2, averageEfficiency: 0.9 })
  })

  it('drops out-of-order and duplicate events', () => {
    const state: EventDedupState = { lastTimestamp: 1000, seenIds: ['e1'] }
    expect(isStaleEvent(alertEvent(900, 'e2'), state)).toBe(true)
    expect(isStaleEvent(alertEvent(1000, 'e1'), state)).toBe(true)
    expect(isStaleEvent(alertEvent(2000, 'e2'), state)).toBe(false)
    expect(nextDedupState(alertEvent(2000, 'e2'), state)).toMatchObject({ lastTimestamp: 2000 })
  })
})
