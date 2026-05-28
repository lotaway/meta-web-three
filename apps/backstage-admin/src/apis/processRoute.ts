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

// 创建工艺路线
export function createProcessRouteAPI(data: CreateProcessRouteRequest) {
  return http<ProcessRoute>({
    url: '/api/mes/process-route',
    method: 'post',
    data,
  })
}

// 更新工艺路线
export function updateProcessRouteAPI(id: number, data: UpdateProcessRouteRequest) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}`,
    method: 'put',
    data,
  })
}

// 删除工艺路线
export function deleteProcessRouteAPI(id: number) {
  return http({
    url: `/api/mes/process-route/${id}`,
    method: 'delete',
  })
}

// 根据ID获取工艺路线
export function getProcessRouteByIdAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}`,
    method: 'get',
  })
}

// 根据路线编码获取工艺路线
export function getProcessRouteByCodeAPI(routeCode: string) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/code/${routeCode}`,
    method: 'get',
  })
}

// 根据产品编码获取工艺路线列表
export function getProcessRouteByProductAPI(productCode: string) {
  return http<ProcessRoute[]>({
    url: `/api/mes/process-route/product/${productCode}`,
    method: 'get',
  })
}

// 获取工艺路线列表（可按状态筛选）
export function getProcessRouteListAPI(status?: string) {
  return http<ProcessRoute[]>({
    url: '/api/mes/process-route',
    method: 'get',
    params: status ? { status } : {},
  })
}

// 激活工艺路线
export function activateProcessRouteAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}/activate`,
    method: 'post',
  })
}

// 归档工艺路线
export function archiveProcessRouteAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}/archive`,
    method: 'post',
  })
}

// 验证工艺路线
export function validateProcessRouteAPI(id: number) {
  return http<ProcessRoute>({
    url: `/api/mes/process-route/${id}/validate`,
    method: 'post',
  })
}

// 获取下一道工序
export function getNextStepAPI(id: number, stepNo: number) {
  return http<ProcessStep>({
    url: `/api/mes/process-route/${id}/next-step/${stepNo}`,
    method: 'get',
  })
}

// 获取首道工序
export function getFirstStepAPI(id: number) {
  return http<ProcessStep>({
    url: `/api/mes/process-route/${id}/first-step`,
    method: 'get',
  })
}
