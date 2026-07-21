import http from '@/utils/http'

export interface ProductionOrder {
  id: number
  orderCode: string
  productCode: string
  productName: string
  quantityPlanned: number
  quantityCompleted: number
  status: 'PENDING' | 'SCHEDULED' | 'IN_PROGRESS' | 'PAUSED' | 'COMPLETED' | 'CANCELLED'
  priority: 'LOW' | 'NORMAL' | 'HIGH' | 'URGENT'
  workshopCode: string
  productionLineCode: string
  progressPercentage: number
  plannedStartTime: string
  plannedEndTime: string
  actualStartTime: string
  actualEndTime: string
  orderType: string
  customerName: string
  notes: string
  createdAt: string
  updatedAt: string
}

export interface WorkStation {
  id: number
  stationCode: string
  stationName: string
  stationType: string
  workshopCode: string
  productionLineCode: string
  status: 'IDLE' | 'OPERATING' | 'MAINTENANCE' | 'BREAKDOWN' | 'OFFLINE'
  capacity: number
  currentLoad: number
  efficiency: number
  currentOperator: string
  currentOrderCode: string
  createdAt: string
}

export interface WorkStationBinding {
  id: number
  workstationCode: string
  bindingType: 'EQUIPMENT' | 'TOOL' | 'PERSONNEL'
  targetCode: string
  targetName: string
  targetType: string
  quantity: number
  isPrimary: boolean
  status: 'ACTIVE' | 'INACTIVE'
  createdAt: string
}

// Production Orders
export function listOrdersAPI() {
  return http<ProductionOrder[]>({ url: '/api/v1/production/orders', method: 'get' })
}

export function getOrderAPI(id: number) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}`, method: 'get' })
}

export function createOrderAPI(data: Partial<ProductionOrder>) {
  return http<ProductionOrder>({ url: '/api/v1/production/orders', method: 'post', data })
}

export function scheduleOrderAPI(id: number, productionLineCode: string) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}/schedule`, method: 'post', data: { productionLineCode } })
}

export function startProductionAPI(id: number) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}/start`, method: 'post' })
}

export function pauseProductionAPI(id: number) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}/pause`, method: 'post' })
}

export function resumeProductionAPI(id: number) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}/resume`, method: 'post' })
}

export function completeProductionAPI(id: number) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}/complete`, method: 'post' })
}

export function cancelOrderAPI(id: number) {
  return http<ProductionOrder>({ url: `/api/v1/production/orders/${id}/cancel`, method: 'post' })
}

// Work Stations
export function listStationsAPI() {
  return http<WorkStation[]>({ url: '/api/v1/production/stations', method: 'get' })
}

export function getStationAPI(id: number) {
  return http<WorkStation>({ url: `/api/v1/production/stations/${id}`, method: 'get' })
}

export function getAvailableStationsAPI() {
  return http<WorkStation[]>({ url: '/api/v1/production/stations/available', method: 'get' })
}

export function createStationAPI(data: Partial<WorkStation>) {
  return http<WorkStation>({ url: '/api/v1/production/stations', method: 'post', data })
}

export function assignOrderToStationAPI(code: string, orderCode: string) {
  return http<WorkStation>({ url: `/api/v1/production/stations/${code}/assign`, method: 'post', data: { orderCode } })
}

// Station Bindings
export function getStationBindingsAPI(stationCode: string) {
  return http<WorkStationBinding[]>({ url: `/api/v1/production/station-bindings/workstation/${stationCode}`, method: 'get' })
}

export function bindEquipmentAPI(stationCode: string, equipmentCode: string, equipmentName: string, equipmentType: string) {
  return http<WorkStationBinding>({ url: '/api/v1/production/station-bindings/equipment', method: 'post', data: { workstationCode: stationCode, equipmentCode, equipmentName, equipmentType } })
}

export function bindToolAPI(stationCode: string, toolCode: string, toolName: string, toolType: string) {
  return http<WorkStationBinding>({ url: '/api/v1/production/station-bindings/tool', method: 'post', data: { workstationCode: stationCode, toolCode, toolName, toolType } })
}

export function bindPersonnelAPI(stationCode: string, personnelCode: string, personnelName: string, personnelType: string) {
  return http<WorkStationBinding>({ url: '/api/v1/production/station-bindings/personnel', method: 'post', data: { workstationCode: stationCode, personnelCode, personnelName, personnelType } })
}

export function unbindStationAPI(id: number) {
  return http<void>({ url: `/api/v1/production/station-bindings/${id}/unbind`, method: 'post' })
}

export function deleteBindingAPI(id: number) {
  return http<void>({ url: `/api/v1/production/station-bindings/${id}`, method: 'delete' })
}
