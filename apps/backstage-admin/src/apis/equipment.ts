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

// Maintenance Plan
export interface MaintenancePlan {
  id?: number
  equipmentId: number
  equipmentCode?: string
  equipmentName?: string
  planCode: string
  planName: string
  maintenanceType: 'PREVENTIVE' | 'PREDICTIVE' | 'CORRECTIVE'
  intervalDays: number
  intervalHours: number
  nextExecutionTime: string
  lastExecutionTime?: string
  estimatedDuration: number
  assignedTo?: string
  status: 'ACTIVE' | 'INACTIVE' | 'COMPLETED'
  description?: string
  checkItems?: string
  createdAt?: string
  updatedAt?: string
}

export interface CreateMaintenancePlanRequest {
  equipmentId: number
  planCode: string
  planName: string
  maintenanceType: 'PREVENTIVE' | 'PREDICTIVE' | 'CORRECTIVE'
  intervalDays: number
  intervalHours: number
  nextExecutionTime: string
  estimatedDuration: number
  assignedTo?: string
  description?: string
  checkItems?: string
}

export function getMaintenancePlanListAPI(params?: {
  equipmentId?: number
  maintenanceType?: string
  status?: string
}) {
  return http<MaintenancePlan[]>({
    url: '/api/mes/equipment/maintenance-plans',
    method: 'get',
    params,
  })
}

export function getMaintenancePlanByIdAPI(id: number) {
  return http<MaintenancePlan>({
    url: `/api/mes/equipment/maintenance-plans/${id}`,
    method: 'get',
  })
}

export function createMaintenancePlanAPI(data: CreateMaintenancePlanRequest) {
  return http<MaintenancePlan>({
    url: '/api/mes/equipment/maintenance-plans',
    method: 'post',
    data,
  })
}

export function updateMaintenancePlanAPI(id: number, data: Partial<CreateMaintenancePlanRequest>) {
  return http<MaintenancePlan>({
    url: `/api/mes/equipment/maintenance-plans/${id}`,
    method: 'put',
    data,
  })
}

export function deleteMaintenancePlanAPI(id: number) {
  return http({
    url: `/api/mes/equipment/maintenance-plans/${id}`,
    method: 'delete',
  })
}

export function executeMaintenancePlanAPI(id: number, result?: string) {
  return http<MaintenancePlan>({
    url: `/api/mes/equipment/maintenance-plans/${id}/execute`,
    method: 'post',
    data: { result },
  })
}

// Equipment Checklist
export interface EquipmentChecklist {
  id?: number
  equipmentId: number
  equipmentCode?: string
  equipmentName?: string
  checkCode: string
  checkType: 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'QUARTERLY' | 'ANNUAL'
  checkTime: string
  checkerId?: string
  checkerName?: string
  status: 'PENDING' | 'COMPLETED' | 'ABNORMAL'
  result?: string
  abnormalItems?: string
  remarks?: string
  createdAt?: string
  updatedAt?: string
}

export interface CreateChecklistRequest {
  equipmentId: number
  checkCode: string
  checkType: 'DAILY' | 'WEEKLY' | 'MONTHLY' | 'QUARTERLY' | 'ANNUAL'
  checkTime: string
  checkerId?: string
  checkerName?: string
  status?: 'PENDING' | 'COMPLETED' | 'ABNORMAL'
  result?: string
  abnormalItems?: string
  remarks?: string
}

export function getChecklistListAPI(params?: {
  equipmentId?: number
  checkType?: string
  status?: string
  startDate?: string
  endDate?: string
}) {
  return http<EquipmentChecklist[]>({
    url: '/api/mes/equipment/checklists',
    method: 'get',
    params,
  })
}

export function getChecklistByIdAPI(id: number) {
  return http<EquipmentChecklist>({
    url: `/api/mes/equipment/checklists/${id}`,
    method: 'get',
  })
}

export function createChecklistAPI(data: CreateChecklistRequest) {
  return http<EquipmentChecklist>({
    url: '/api/mes/equipment/checklists',
    method: 'post',
    data,
  })
}

export function updateChecklistAPI(id: number, data: Partial<CreateChecklistRequest>) {
  return http<EquipmentChecklist>({
    url: `/api/mes/equipment/checklists/${id}`,
    method: 'put',
    data,
  })
}

export function deleteChecklistAPI(id: number) {
  return http({
    url: `/api/mes/equipment/checklists/${id}`,
    method: 'delete',
  })
}

export function completeChecklistAPI(id: number, data: {
  result?: string
  abnormalItems?: string
  remarks?: string
}) {
  return http<EquipmentChecklist>({
    url: `/api/mes/equipment/checklists/${id}/complete`,
    method: 'post',
    data,
  })
}