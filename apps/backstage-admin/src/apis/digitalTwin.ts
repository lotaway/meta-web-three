import http from '@/utils/http'

export interface DigitalTwinDevice {
  id: number
  deviceCode: string
  deviceName: string
  deviceType: string
  workshopId: string
  productionLineId: string
  status: string
  positionX: number
  positionY: number
  positionZ: number
  rotation: number
  efficiency: number
  lastHeartbeat: string
  createdAt: string
  updatedAt: string
}

export interface Workshop {
  id: number
  workshopCode: string
  workshopName: string
  description: string
  createdAt: string
}

export interface ProductionLine {
  id: number
  lineCode: string
  lineName: string
  workshopId: string
  capacity: number
  createdAt: string
}

export interface Alert {
  id: number
  alertCode: string
  deviceCode: string
  workshopId: string
  level: string
  type: string
  title: string
  description: string
  status: string
  solution: string
  acknowledgedBy: string
  resolvedBy: string
  occurredAt: string
  acknowledgedAt: string
  resolvedAt: string
  createdAt: string
}

export interface AlertRule {
  id: number
  ruleCode: string
  ruleName: string
  description: string
  deviceType: string
  deviceCode: string
  workshopId: string
  metricType: string
  operator: string
  thresholdValue: number
  durationSeconds: number
  level: string
  alertType: string
  titleTemplate: string
  descriptionTemplate: string
  enabled: boolean
  createdAt: string
  updatedAt: string
}

export interface StatsSummary {
  onlineDeviceCount: number
  activeAlertCount: number
  averageEfficiency: number
}

export function getDevicesAPI() {
  return http<DigitalTwinDevice[]>({ url: '/api/digital-twin/devices', method: 'get' })
}

export function getDeviceByIdAPI(id: number) {
  return http<DigitalTwinDevice>({ url: `/api/digital-twin/device/${id}`, method: 'get' })
}

export function registerDeviceAPI(data: {
  deviceCode: string; deviceName: string; deviceType: string;
  workshopId: string; productionLineId: string;
}) {
  return http<{ deviceId: number }>({ url: '/api/digital-twin/device', method: 'post', data })
}

export function updateDeviceStatusAPI(deviceCode: string, status: string) {
  return http<void>({ url: `/api/digital-twin/device/${deviceCode}/status`, method: 'post', data: { status } })
}

export function updateDevicePositionAPI(deviceCode: string, x: number, y: number, z: number, rotation?: number) {
  return http<void>({ url: `/api/digital-twin/device/${deviceCode}/position`, method: 'post', data: { x, y, z, rotation } })
}

export function deviceHeartbeatAPI(deviceCode: string) {
  return http<void>({ url: `/api/digital-twin/device/${deviceCode}/heartbeat`, method: 'post' })
}

export function getWorkshopDevicesAPI(workshopId: string) {
  return http<DigitalTwinDevice[]>({ url: `/api/digital-twin/workshop/${workshopId}/devices`, method: 'get' })
}

export function getOnlineDevicesAPI() {
  return http<DigitalTwinDevice[]>({ url: '/api/digital-twin/devices/online', method: 'get' })
}

export function createWorkshopAPI(data: { workshopCode: string; workshopName: string; description?: string }) {
  return http<{ workshopId: number }>({ url: '/api/digital-twin/workshop', method: 'post', data })
}

export function getWorkshopsAPI() {
  return http<Workshop[]>({ url: '/api/digital-twin/workshops', method: 'get' })
}

export function createProductionLineAPI(data: { lineCode: string; lineName: string; workshopId: string; capacity: number }) {
  return http<{ lineId: number }>({ url: '/api/digital-twin/production-line', method: 'post', data })
}

export function getProductionLinesAPI() {
  return http<ProductionLine[]>({ url: '/api/digital-twin/production-lines', method: 'get' })
}

export function createAlertAPI(data: { deviceCode: string; workshopId: string; level: string; type: string; title: string; description: string }) {
  return http<{ alertId: number }>({ url: '/api/digital-twin/alert', method: 'post', data })
}

export function acknowledgeAlertAPI(id: number, acknowledgedBy: string) {
  return http<void>({ url: `/api/digital-twin/alert/${id}/acknowledge`, method: 'post', data: { acknowledgedBy } })
}

export function resolveAlertAPI(id: number, solution: string, resolvedBy: string) {
  return http<void>({ url: `/api/digital-twin/alert/${id}/resolve`, method: 'post', data: { solution, resolvedBy } })
}

export function getActiveAlertsAPI() {
  return http<Alert[]>({ url: '/api/digital-twin/alerts/active', method: 'get' })
}

export function getStatsSummaryAPI() {
  return http<StatsSummary>({ url: '/api/digital-twin/stats/summary', method: 'get' })
}

export function getAlertRulesAPI(enabled?: boolean, deviceType?: string) {
  return http<AlertRule[]>({ url: '/api/alert-rules', method: 'get', params: { enabled, deviceType } })
}

export function getAlertRuleByIdAPI(id: number) {
  return http<AlertRule>({ url: `/api/alert-rules/${id}`, method: 'get' })
}

export function createAlertRuleAPI(data: Record<string, unknown>) {
  return http<AlertRule>({ url: '/api/alert-rules', method: 'post', data })
}

export function updateAlertRuleAPI(id: number, data: Record<string, unknown>) {
  return http<AlertRule>({ url: `/api/alert-rules/${id}`, method: 'put', data })
}

export function enableAlertRuleAPI(id: number) {
  return http<Record<string, unknown>>({ url: `/api/alert-rules/${id}/enable`, method: 'put' })
}

export function disableAlertRuleAPI(id: number) {
  return http<Record<string, unknown>>({ url: `/api/alert-rules/${id}/disable`, method: 'put' })
}

export function deleteAlertRuleAPI(id: number) {
  return http<Record<string, unknown>>({ url: `/api/alert-rules/${id}`, method: 'delete' })
}
