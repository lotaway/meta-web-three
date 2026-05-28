import http from '@/utils/http'

export type EquipmentStatus = 'IDLE' | 'RUNNING' | 'BREAKDOWN' | 'MAINTENANCE' | 'OFFLINE' | 'ONLINE' | 'WARNING' | 'ERROR'

export interface Equipment {
  id?: number
  equipmentCode: string
  equipmentName: string
  equipmentTypeId?: number
  equipmentTypeCode: string
  workshopId?: string
  workstationId?: string
  statusCode?: string
  status: EquipmentStatus
  statusConfigId?: number
  currentTaskNo?: string
  utilizationRate?: number
  todayOutput?: number
  lastMaintenanceTime?: string
  nextMaintenanceTime?: string
  totalRunningSeconds?: number
  totalIdleSeconds?: number
  totalDowntimeSeconds?: number
  digitalTwinDeviceCode?: string
  positionX?: number
  positionY?: number
  positionZ?: number
  rotationY?: number
  ipAddress?: string
  macAddress?: string
  mqttTopic?: string
  lastHeartbeat?: string
  createdAt?: string
  updatedAt?: string
}

export interface CreateEquipmentRequest {
  equipmentCode: string
  equipmentName: string
  equipmentTypeId?: number
  equipmentTypeCode: string
  workshopId?: string
  workstationId?: string
  positionX?: number
  positionY?: number
  positionZ?: number
  ipAddress?: string
  macAddress?: string
}

export interface UpdateEquipmentRequest {
  equipmentName: string
  equipmentTypeId?: number
  equipmentTypeCode: string
  workshopId?: string
  workstationId?: string
  positionX?: number
  positionY?: number
  positionZ?: number
  ipAddress?: string
  macAddress?: string
}

export interface EquipmentStatusVO {
  equipmentId: number
  status: EquipmentStatus
  currentTaskNo?: string
  utilizationRate?: number
  todayOutput?: number
  totalRunningSeconds?: number
  totalIdleSeconds?: number
  totalDowntimeSeconds?: number
}

export function createEquipmentAPI(data: CreateEquipmentRequest) {
  return http<Equipment>({
    url: '/api/mes/equipment',
    method: 'post',
    data,
  })
}

export function updateEquipmentAPI(id: number, data: UpdateEquipmentRequest) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}`,
    method: 'put',
    data,
  })
}

export function deleteEquipmentAPI(id: number) {
  return http({
    url: `/api/mes/equipment/${id}`,
    method: 'delete',
  })
}

export function getEquipmentByIdAPI(id: number) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}`,
    method: 'get',
  })
}

export function getEquipmentListAPI(params?: { 
  workshopId?: string
  status?: EquipmentStatus
  equipmentTypeCode?: string
}) {
  return http<Equipment[]>({
    url: '/api/mes/equipment',
    method: 'get',
    params,
  })
}

export function startTaskAPI(id: number, taskNo: string) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/start-task`,
    method: 'post',
    data: { taskNo },
  })
}

export function completeTaskAPI(id: number) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/complete-task`,
    method: 'post',
  })
}

export function reportBreakdownAPI(id: number, reason?: string) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/report-breakdown`,
    method: 'post',
    data: { reason },
  })
}

export function repairEquipmentAPI(id: number) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/repair`,
    method: 'post',
  })
}

export function startMaintenanceAPI(id: number) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/start-maintenance`,
    method: 'post',
  })
}

export function completeMaintenanceAPI(id: number) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/complete-maintenance`,
    method: 'post',
  })
}

export function getEquipmentStatusAPI(id: number) {
  return http<EquipmentStatusVO>({
    url: `/api/mes/equipment/${id}/status`,
    method: 'get',
  })
}

export function bindWorkstationAPI(id: number, workstationId: string) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/bind-workstation`,
    method: 'post',
    data: { workstationId },
  })
}

export function unbindWorkstationAPI(id: number) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/unbind-workstation`,
    method: 'post',
  })
}

export function bindDigitalTwinAPI(id: number, deviceCode: string) {
  return http<Equipment>({
    url: `/api/mes/equipment/${id}/bind-digital-twin`,
    method: 'post',
    data: { deviceCode },
  })
}

export function getOEEAPI(id: number, params?: {
  plannedProductionTime?: number
  idealCycleTime?: number
  goodProductCount?: number
}) {
  return http<{ utilizationRate: number }>({
    url: `/api/mes/equipment/${id}/oee`,
    method: 'get',
    params,
  })
}