import http from '@/utils/http'

export type TaskStatus = 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED' | 'ON_HOLD' | 'QUALITY_CHECK' | 'SCRAPPED'

export interface ProductionTask {
  id?: number
  taskNo: string
  workOrderId: number
  workOrderNo?: string
  workstationId?: string
  processCode?: string
  processName?: string
  quantity: number
  completedQuantity?: number
  qualifiedQuantity?: number
  defectiveQuantity?: number
  operatorId?: string
  operatorName?: string
  status: TaskStatus
  startTime?: string
  endTime?: string
  estimatedDurationMinutes?: number
  actualDurationMinutes?: number
  extensionFields?: Record<string, string>
  createdAt?: string
  updatedAt?: string
}

export interface CreateTaskRequest {
  taskNo: string
  workOrderId: number
  workstationId?: string
  processCode?: string
  quantity: number
  operatorId?: string
}

export interface UpdateTaskRequest {
  workstationId?: string
  processCode?: string
  quantity?: number
  operatorId?: string
}

export interface CompleteTaskRequest {
  qualified: number
  defective: number
}

export function createTaskAPI(data: CreateTaskRequest) {
  return http<ProductionTask>({
    url: '/api/mes/production-tasks',
    method: 'post',
    data,
  })
}

export function updateTaskAPI(id: number, data: UpdateTaskRequest) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}`,
    method: 'put',
    data,
  })
}

export function deleteTaskAPI(id: number) {
  return http({
    url: `/api/mes/production-tasks/${id}`,
    method: 'delete',
  })
}

export function getTaskByIdAPI(id: number) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}`,
    method: 'get',
  })
}

export function getTaskByTaskNoAPI(taskNo: string) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/no/${taskNo}`,
    method: 'get',
  })
}

export function getTaskListAPI(params?: {
  workOrderId?: number
  status?: TaskStatus
  workstationId?: string
}) {
  return http<ProductionTask[]>({
    url: '/api/mes/production-tasks',
    method: 'get',
    params,
  })
}

export function getTasksByWorkOrderAPI(workOrderId: number) {
  return http<ProductionTask[]>({
    url: `/api/mes/production-tasks/work-order/${workOrderId}`,
    method: 'get',
  })
}

export function getTasksByStatusAPI(status: TaskStatus) {
  return http<ProductionTask[]>({
    url: `/api/mes/production-tasks/status/${status}`,
    method: 'get',
  })
}

export function getTasksByWorkstationAPI(workstationId: string) {
  return http<ProductionTask[]>({
    url: `/api/mes/production-tasks/workstation/${workstationId}`,
    method: 'get',
  })
}

export function startTaskAPI(id: number) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}/start`,
    method: 'post',
  })
}

export function completeTaskAPI(id: number, data: CompleteTaskRequest) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}/complete`,
    method: 'post',
    data,
  })
}

export function passQualityCheckAPI(id: number) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}/quality/pass`,
    method: 'post',
  })
}

export function failQualityCheckAPI(id: number) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}/quality/fail`,
    method: 'post',
  })
}

export function scrapTaskAPI(id: number) {
  return http<ProductionTask>({
    url: `/api/mes/production-tasks/${id}/scrap`,
    method: 'post',
  })
}

export function getExtensionFieldValuesAPI(id: number) {
  return http<Array<{ fieldName: string; fieldValue: string }>>({
    url: `/api/mes/production-tasks/${id}/extension-values`,
    method: 'get',
  })
}

export function setExtensionFieldValuesAPI(id: number, fieldValues: Record<string, string>) {
  return http({
    url: `/api/mes/production-tasks/${id}/extension-values`,
    method: 'post',
    data: fieldValues,
  })
}

export function countTasksByWorkOrderAPI(workOrderId: number) {
  return http<{ count: number }>({
    url: `/api/mes/production-tasks/work-order/${workOrderId}/count`,
    method: 'get',
  })
}

export function countTasksByStatusAPI(status: TaskStatus) {
  return http<{ count: number }>({
    url: `/api/mes/production-tasks/status/${status}/count`,
    method: 'get',
  })
}
