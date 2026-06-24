import http from '@/utils/http'

export interface TelemetryMetric {
  metricCode: string
  metricName: string
  value: number
  unit: string
  quality?: string
  upperLimit?: number
  lowerLimit?: number
}

export interface TelemetryRecord {
  id?: number
  equipmentCode: string
  topic: string
  collectTime: string
  metrics: TelemetryMetric[]
  createdAt?: string
}

export interface DeviceCommand {
  id?: number
  commandCode: string
  equipmentCode: string
  commandType: CommandType
  payload: string
  status: CommandStatus
  createdBy: string
  createdAt?: string
  executedAt?: string
  resultMessage?: string
}

export type CommandType = 'START' | 'STOP' | 'RESET' | 'SET_PARAMETER' | 'CALIBRATE' | 'SET_SPEED' | 'SET_TEMPERATURE' | 'CUSTOM'
export type CommandStatus = 'PENDING' | 'SENT' | 'DELIVERED' | 'EXECUTED' | 'FAILED' | 'TIMEOUT'

export function ingestTelemetryAPI(data: {
  equipmentCode: string
  topic: string
  payload: string
  collectTime?: string
}) {
  return http<TelemetryRecord>({ url: '/api/mes/scada/telemetry/ingest', method: 'post', data })
}

export function getTelemetryListAPI(equipmentCode: string, limit?: number) {
  return http<TelemetryRecord[]>({ url: `/api/mes/scada/telemetry/${equipmentCode}`, method: 'get', params: { limit } })
}

export function getTelemetryByRangeAPI(equipmentCode: string, start: string, end: string) {
  return http<TelemetryRecord[]>({ url: `/api/mes/scada/telemetry/${equipmentCode}/range`, method: 'get', params: { start, end } })
}

export function dispatchCommandAPI(data: {
  equipmentCode: string
  commandType: CommandType
  payload: string
  createdBy: string
}) {
  return http<DeviceCommand>({ url: '/api/mes/scada/commands', method: 'post', data })
}

export function getCommandListAPI(equipmentCode: string, status?: CommandStatus) {
  return http<DeviceCommand[]>({ url: `/api/mes/scada/commands/${equipmentCode}`, method: 'get', params: status ? { status } : {} })
}

export function getCommandStatusAPI(commandCode: string) {
  return http<DeviceCommand>({ url: `/api/mes/scada/commands/status/${commandCode}`, method: 'get' })
}

export interface DashboardMetrics {
  totalEquipment: number
  onlineEquipment: number
  runningEquipment: number
  idleEquipment: number
  warningEquipment: number
  errorEquipment: number
  offlineEquipment: number
  avgOee: number
  todayOutput: number
  activeAlerts: number
  pendingWorkOrders: number
}

export interface EquipmentStatusSummary {
  equipmentId: number
  equipmentCode: string
  equipmentName: string
  status: string
  oee: number
  todayOutput: number
  lastHeartbeat: string
  currentTaskNo: string
}

export interface AlertSummary {
  id: number
  eventNo: string
  levelCode: string
  levelName: string
  equipmentId: string
  equipmentCode: string
  equipmentName: string
  description: string
  status: string
  occurredAt: string
  reporterName: string
}

export interface ProductionStats {
  totalWorkOrders: number
  inProgressWorkOrders: number
  completedWorkOrders: number
  pendingWorkOrders: number
  todayOutput: number
  todayPlannedOutput: number
  totalTasks: number
  completedTasks: number
  completionRate: number
}

export function getDashboardMetricsAPI(workshopId?: string) {
  return http<DashboardMetrics>({ url: '/api/mes/scada/dashboard/metrics', method: 'get', params: workshopId ? { workshopId } : {} })
}

export function getEquipmentStatusSummaryAPI(workshopId?: string) {
  return http<EquipmentStatusSummary[]>({ url: '/api/mes/scada/dashboard/equipment', method: 'get', params: workshopId ? { workshopId } : {} })
}

export function getActiveAlertsAPI(workshopId?: string) {
  return http<AlertSummary[]>({ url: '/api/mes/scada/dashboard/alerts', method: 'get', params: workshopId ? { workshopId } : {} })
}

export function getProductionStatsAPI(workshopId?: string) {
  return http<ProductionStats>({ url: '/api/mes/scada/dashboard/production', method: 'get', params: workshopId ? { workshopId } : {} })
}

export interface Workshop {
  id: string
  workshopCode: string
  workshopName: string
  description?: string
}

export function getWorkshopListAPI() {
  return http<Workshop[]>({ url: '/api/digital-twin/workshops', method: 'get' })
}
