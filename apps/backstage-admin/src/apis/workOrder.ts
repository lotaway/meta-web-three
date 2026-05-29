import http from '@/utils/http'

export type WorkOrderStatus = 'DRAFT' | 'RELEASED' | 'IN_PROGRESS' | 'PAUSED' | 'COMPLETED' | 'CANCELLED'
export type WorkOrderPriority = 'LOW' | 'NORMAL' | 'HIGH' | 'URGENT'
export type WorkOrderType = 'NORMAL' | 'REWORK' | 'REPAIR' | 'SAMPLE'
export type SplitType = 'BY_BOM' | 'BY_PROCESS' | 'MANUAL'

export interface WorkOrder {
  id?: number
  workOrderNo: string
  productCode: string
  productName: string
  quantity: number
  completedQuantity?: number
  status: WorkOrderStatus
  statusCode?: string
  typeCode?: WorkOrderType
  priority?: WorkOrderPriority
  workshopId?: string
  workshopName?: string
  processRouteId?: string
  processRouteName?: string
  parentWorkOrderId?: number
  splitRuleId?: number
  splitSequence?: number
  splitType?: SplitType
  plannedStartTime?: string
  plannedEndTime?: string
  actualStartTime?: string
  actualEndTime?: string
  completionRate?: number
  createdAt?: string
  updatedAt?: string
}

export interface CreateWorkOrderRequest {
  workOrderNo: string
  productCode: string
  productName: string
  quantity: number
  workshopId?: string
  processRouteId?: string
  typeCode?: WorkOrderType
}

export interface UpdateWorkOrderRequest {
  productCode?: string
  productName?: string
  quantity?: number
  workshopId?: string
  processRouteId?: string
  priority?: WorkOrderPriority
  plannedStartTime?: string
  plannedEndTime?: string
}

export interface CancelWorkOrderRequest {
  reason?: string
}

export interface UpdateProgressRequest {
  quantity: number
}

export interface SplitWorkOrderRequest {
  splitType: SplitType
  splitCount: number
}

export function createWorkOrderAPI(data: CreateWorkOrderRequest) {
  return http<WorkOrder>({
    url: '/api/mes/work-orders',
    method: 'post',
    data,
  })
}

export function updateWorkOrderAPI(id: number, data: UpdateWorkOrderRequest) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}`,
    method: 'put',
    data,
  })
}

export function deleteWorkOrderAPI(id: number) {
  return http({
    url: `/api/mes/work-orders/${id}`,
    method: 'delete',
  })
}

export function getWorkOrderByIdAPI(id: number) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}`,
    method: 'get',
  })
}

export function getWorkOrderByNoAPI(workOrderNo: string) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/no/${workOrderNo}`,
    method: 'get',
  })
}

export function getWorkOrderListAPI(params?: {
  status?: WorkOrderStatus
  workshopId?: string
  productCode?: string
}) {
  return http<WorkOrder[]>({
    url: '/api/mes/work-orders',
    method: 'get',
    params,
  })
}

export function getWorkOrdersByStatusAPI(status: WorkOrderStatus) {
  return http<WorkOrder[]>({
    url: `/api/mes/work-orders/status/${status}`,
    method: 'get',
  })
}

export function getWorkOrdersByWorkshopAPI(workshopId: string) {
  return http<WorkOrder[]>({
    url: `/api/mes/work-orders/workshop/${workshopId}`,
    method: 'get',
  })
}

export function getWorkOrdersByProductCodeAPI(productCode: string) {
  return http<WorkOrder[]>({
    url: `/api/mes/work-orders/product/${productCode}`,
    method: 'get',
  })
}

export function getChildWorkOrdersAPI(parentWorkOrderId: number) {
  return http<WorkOrder[]>({
    url: `/api/mes/work-orders/parent/${parentWorkOrderId}`,
    method: 'get',
  })
}

export function releaseWorkOrderAPI(id: number) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/release`,
    method: 'post',
  })
}

export function startWorkOrderAPI(id: number) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/start`,
    method: 'post',
  })
}

export function pauseWorkOrderAPI(id: number) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/pause`,
    method: 'post',
  })
}

export function resumeWorkOrderAPI(id: number) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/resume`,
    method: 'post',
  })
}

export function completeWorkOrderAPI(id: number) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/complete`,
    method: 'post',
  })
}

export function cancelWorkOrderAPI(id: number, data?: CancelWorkOrderRequest) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/cancel`,
    method: 'post',
    data,
  })
}

export function updateProgressAPI(id: number, data: UpdateProgressRequest) {
  return http<WorkOrder>({
    url: `/api/mes/work-orders/${id}/progress`,
    method: 'post',
    data,
  })
}

export function splitWorkOrderAPI(id: number, data: SplitWorkOrderRequest) {
  return http<WorkOrder[]>({
    url: `/api/mes/work-orders/${id}/split`,
    method: 'post',
    data,
  })
}