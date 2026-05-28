import http from '@/utils/http'

export interface ProcessStep {
  stepNo: number
  processCode: string
  processName: string
  workstationId?: string
  standardTime?: number
  qualityCheckpoint?: string
  predecessorStepNo?: number
  successorStepNo?: number
}

export interface ProcessRoute {
  id?: number
  routeCode: string
  routeName: string
  productCode: string
  version?: number
  status?: 'DRAFT' | 'ACTIVE' | 'ARCHIVED'
  steps?: ProcessStep[]
  createdAt?: string
  updatedAt?: string
  validationMessage?: string
  validationResult?: boolean
}

export interface CreateProcessRouteRequest {
  routeCode: string
  routeName: string
  productCode: string
  steps?: ProcessStep[]
}

export interface UpdateProcessRouteRequest {
  routeName: string
  productCode: string
  steps?: ProcessStep[]
}

export function createProcessRouteAPI(data: CreateProcessRouteRequest) {
  return http<ProcessRoute>({
    url: '/api/mes/process-route',
    method: 'post',
    data,
  })
}

export function updateProcessRouteAPI(id: number, data: UpdateProcessRouteRequest) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}`,
    method: 'put',
    data,
  })
}

export function deleteProcessRouteAPI(id: number) {
  return http({
    url: `/api/mes/process-route/${id}`,
    method: 'delete',
  })
}

export function getProcessRouteByIdAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}`,
    method: 'get',
  })
}

export function getProcessRouteByCodeAPI(routeCode: string) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/code/${routeCode}`,
    method: 'get',
  })
}

export function getProcessRouteByProductAPI(productCode: string) {
  return http<ProcessRoute[]>({
    url: `/api/mes/process-route/product/${productCode}`,
    method: 'get',
  })
}

export function getProcessRouteListAPI(status?: string) {
  return http<ProcessRoute[]>({
    url: '/api/mes/process-route',
    method: 'get',
    params: status ? { status } : {},
  })
}

export function activateProcessRouteAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}/activate`,
    method: 'post',
  })
}

export function archiveProcessRouteAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}/archive`,
    method: 'post',
  })
}

export function validateProcessRouteAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}/validate`,
    method: 'post',
  })
}

export function getNextStepAPI(id: number, stepNo: number) {
  return http<ProcessStep>({
    url: `/api/mes/process-route/${id}/next-step/${stepNo}`,
    method: 'get',
  })
}

export function getFirstStepAPI(id: number) {
  return http<ProcessStep>({
    url: `/api/mes/process-route/${id}/first-step`,
    method: 'get',
  })
}
