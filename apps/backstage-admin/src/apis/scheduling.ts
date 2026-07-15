import http from '@/utils/http'

export type ScheduleOrderStatus = 'PENDING' | 'SCHEDULED' | 'IN_PROGRESS' | 'COMPLETED' | 'DELAYED' | 'CANCELLED'
export type SchedulePriority = 'LOW' | 'NORMAL' | 'HIGH' | 'URGENT'
export type ScheduleOpStatus = 'PENDING' | 'SCHEDULED' | 'IN_PROGRESS' | 'COMPLETED' | 'BLOCKED'
export type ResourceType = 'EQUIPMENT' | 'WORK_CENTER' | 'LABOR' | 'TOOL'
export type ResourceStatus = 'AVAILABLE' | 'OCCUPIED' | 'MAINTENANCE' | 'OFFLINE'

export interface ScheduleOperation {
  id?: number
  scheduleOrderId?: number
  operationCode: string
  operationName: string
  sequenceNo?: number
  resourceCode: string
  resourceName: string
  setupTimeMinutes: number
  processingTimeMinutes: number
  teardownTimeMinutes?: number
  status?: ScheduleOpStatus
  scheduledStartTime?: string
  scheduledEndTime?: string
}

export interface ScheduleOrder {
  id?: number
  scheduleNo: string
  orderNo: string
  productCode: string
  productName: string
  quantity: number
  completedQuantity?: number
  dueDate?: string
  scheduledStartTime?: string
  scheduledEndTime?: string
  actualStartTime?: string
  actualEndTime?: string
  priority: SchedulePriority
  status: ScheduleOrderStatus
  workshopId: string
  routeCode?: string
  operations?: ScheduleOperation[]
  remark?: string
  createdBy?: string
  createdAt?: string
  updatedAt?: string
}

export interface ScheduleConflict {
  orderNo: string
  productCode: string
  resourceCode: string
  constraintType: string
  description: string
}

export interface ScheduleResult {
  status: 'SUCCESS' | 'PARTIAL' | 'FAILED'
  direction: 'FORWARD' | 'BACKWARD'
  scheduledOrders: ScheduleOrder[]
  conflicts: ScheduleConflict[]
  computedAt: string
  computationTimeMs: number
  totalOrders: number
  scheduledCount: number
  failedCount: number
}

export interface ScheduleResource {
  id?: number
  resourceCode: string
  resourceName: string
  resourceType: ResourceType
  status: ResourceStatus
  workshopId: string
  capacityPerShift?: number
  calendarCode?: string
  description?: string
  createdAt?: string
  updatedAt?: string
}

export interface CreateOrderRequest {
  scheduleNo: string
  orderNo: string
  productCode: string
  productName: string
  quantity: number
  dueDate?: string
  priority: SchedulePriority
  workshopId: string
  routeCode?: string
}

export interface OperationRequest {
  operationCode: string
  operationName: string
  resourceCode: string
  resourceName: string
  setupTimeMinutes: number
  processingTimeMinutes: number
  teardownTimeMinutes?: number
}

export function getScheduleOrderListAPI(params?: {
  workshopId?: string
  status?: ScheduleOrderStatus
}) {
  return http<ScheduleOrder[]>({ url: '/api/mes/scheduling/orders', method: 'get', params })
}

export function getScheduleOrderByIdAPI(id: number) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}`, method: 'get' })
}

export function getOverdueOrdersAPI() {
  return http<ScheduleOrder[]>({ url: '/api/mes/scheduling/orders/overdue', method: 'get' })
}

export function createScheduleOrderAPI(data: CreateOrderRequest) {
  return http<ScheduleOrder>({ url: '/api/mes/scheduling/orders', method: 'post', data })
}

export function addOperationsAPI(id: number, data: OperationRequest[]) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}/operations`, method: 'post', data })
}

export function deleteScheduleOrderAPI(id: number) {
  return http({ url: `/api/mes/scheduling/orders/${id}`, method: 'delete' })
}

export function runForwardScheduleAPI(workshopId: string) {
  return http<ScheduleResult>({ url: '/api/mes/scheduling/forward', method: 'post', params: { workshopId } })
}

export function runBackwardScheduleAPI(workshopId: string) {
  return http<ScheduleResult>({ url: '/api/mes/scheduling/backward', method: 'post', params: { workshopId } })
}

export function rescheduleOrderAPI(id: number) {
  return http<ScheduleResult>({ url: `/api/mes/scheduling/orders/${id}/reschedule`, method: 'post' })
}

export function startOrderAPI(id: number) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}/start`, method: 'post' })
}

export function completeOrderAPI(id: number) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}/complete`, method: 'post' })
}

export function cancelOrderAPI(id: number) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}/cancel`, method: 'post' })
}

export function markDelayedAPI(id: number) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}/delay`, method: 'post' })
}

export function updateProgressAPI(id: number, completedQuantity: number) {
  return http<ScheduleOrder>({ url: `/api/mes/scheduling/orders/${id}/progress`, method: 'put', data: { completedQuantity } })
}

export function getResourceListAPI(params?: { workshopId?: string; resourceType?: ResourceType }) {
  return http<ScheduleResource[]>({ url: '/api/mes/scheduling/resources', method: 'get', params })
}

export function getResourceByIdAPI(id: number) {
  return http<ScheduleResource>({ url: `/api/mes/scheduling/resources/${id}`, method: 'get' })
}

export function createResourceAPI(data: {
  resourceCode: string
  resourceName: string
  resourceType: ResourceType
  workshopId: string
  capacityPerShift?: number
  description?: string
}) {
  return http<ScheduleResource>({ url: '/api/mes/scheduling/resources', method: 'post', data })
}

export function updateResourceAPI(id: number, data: {
  resourceName?: string
  capacityPerShift?: number
  status?: ResourceStatus
  description?: string
}) {
  return http<ScheduleResource>({ url: `/api/mes/scheduling/resources/${id}`, method: 'put', data })
}

export function deleteResourceAPI(id: number) {
  return http({ url: `/api/mes/scheduling/resources/${id}`, method: 'delete' })
}
