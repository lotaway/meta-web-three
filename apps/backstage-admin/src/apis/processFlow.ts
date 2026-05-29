import http from '@/utils/http'

export interface ProcessFlowTemplate {
  id?: number
  templateCode: string
  templateName: string
  description?: string
  version?: number
  flowData?: string
  status?: 'DRAFT' | 'PUBLISHED' | 'ARCHIVED'
  createdBy?: number
  createdAt?: string
  updatedBy?: number
  updatedAt?: string
  deleted?: boolean
}

export interface ProcessFlowTemplateVersion {
  id?: number
  templateId: number
  version: number
  flowData: string
  changeDescription?: string
  createdBy?: number
  createdAt?: string
}

export interface ProcessFlowInstance {
  id?: number
  templateId: number
  templateCode?: string
  templateName?: string
  businessType?: string
  businessKey?: string
  status?: 'RUNNING' | 'COMPLETED' | 'TERMINATED'
  currentNodeId?: string
  flowData?: string
  startedBy?: number
  startedAt?: string
  completedAt?: string
  updatedAt?: string
}

export interface ProcessNodeType {
  id?: number
  typeCode: string
  typeName: string
  category?: string
  icon?: string
  defaultConfig?: string
  createdAt?: string
}

// ========== 流程模板管理 ==========

export function createProcessTemplateAPI(data: ProcessFlowTemplate) {
  return http<ProcessFlowTemplate>({
    url: '/api/mes/process-flow/templates',
    method: 'post',
    data,
  })
}

export function updateProcessTemplateAPI(id: number, data: ProcessFlowTemplate) {
  return http<ProcessFlowTemplate>({
    url: `/api/mes/process-flow/templates/${id}`,
    method: 'put',
    data,
  })
}

export function deleteProcessTemplateAPI(id: number) {
  return http({
    url: `/api/mes/process-flow/templates/${id}`,
    method: 'delete',
  })
}

export function getProcessTemplateAPI(id: number) {
  return http<ProcessFlowTemplate>({
    url: `/api/mes/process-flow/templates/${id}`,
    method: 'get',
  })
}

export function listProcessTemplatesAPI(status?: string) {
  return http<ProcessFlowTemplate[]>({
    url: '/api/mes/process-flow/templates',
    method: 'get',
    params: status ? { status } : {},
  })
}

export function publishProcessTemplateAPI(id: number, userId: number) {
  return http({
    url: `/api/mes/process-flow/templates/${id}/publish`,
    method: 'post',
    params: { userId },
  })
}

export function archiveProcessTemplateAPI(id: number, userId: number) {
  return http({
    url: `/api/mes/process-flow/templates/${id}/archive`,
    method: 'post',
    params: { userId },
  })
}

// ========== 流程模板版本管理 ==========

export function getTemplateVersionHistoryAPI(templateId: number) {
  return http<ProcessFlowTemplateVersion[]>({
    url: `/api/mes/process-flow/templates/${templateId}/versions`,
    method: 'get',
  })
}

export function rollbackToVersionAPI(templateId: number, version: number, userId: number) {
  return http<ProcessFlowTemplate>({
    url: `/api/mes/process-flow/templates/${templateId}/versions/${version}/rollback`,
    method: 'post',
    params: { userId },
  })
}

export function saveTemplateVersionAPI(templateId: number, changeDescription: string) {
  return http({
    url: `/api/mes/process-flow/templates/${templateId}/save-version`,
    method: 'post',
    params: { changeDescription },
  })
}

// ========== 节点类型管理 ==========

export function listNodeTypesAPI(category?: string) {
  return http<ProcessNodeType[]>({
    url: '/api/mes/process-flow/node-types',
    method: 'get',
    params: category ? { category } : {},
  })
}

export function createNodeTypeAPI(data: ProcessNodeType) {
  return http<ProcessNodeType>({
    url: '/api/mes/process-flow/node-types',
    method: 'post',
    data,
  })
}

export function updateNodeTypeAPI(id: number, data: ProcessNodeType) {
  return http<ProcessNodeType>({
    url: `/api/mes/process-flow/node-types/${id}`,
    method: 'put',
    data,
  })
}

export function deleteNodeTypeAPI(id: number) {
  return http({
    url: `/api/mes/process-flow/node-types/${id}`,
    method: 'delete',
  })
}

export function initDefaultNodeTypesAPI() {
  return http({
    url: '/api/mes/process-flow/node-types/init',
    method: 'post',
  })
}

// ========== 流程实例管理 ==========

export function startProcessInstanceAPI(data: {
  templateId: number
  businessType: string
  businessKey: string
  userId: number
}) {
  return http<ProcessFlowInstance>({
    url: '/api/mes/process-flow/instances',
    method: 'post',
    data,
  })
}

export function completeProcessInstanceAPI(id: number, userId: number) {
  return http({
    url: `/api/mes/process-flow/instances/${id}/complete`,
    method: 'post',
    params: { userId },
  })
}

export function terminateProcessInstanceAPI(id: number, userId: number) {
  return http({
    url: `/api/mes/process-flow/instances/${id}/terminate`,
    method: 'post',
    params: { userId },
  })
}

export function getProcessInstanceAPI(id: number) {
  return http<ProcessFlowInstance>({
    url: `/api/mes/process-flow/instances/${id}`,
    method: 'get',
  })
}

export function listProcessInstancesAPI(businessType?: string, status?: string) {
  return http<ProcessFlowInstance[]>({
    url: '/api/mes/process-flow/instances',
    method: 'get',
    params: {
      ...(businessType && { businessType }),
      ...(status && { status }),
    },
  })
}

export function listProcessInstancesByBusinessAPI(businessType: string, businessKey: string) {
  return http<ProcessFlowInstance[]>({
    url: '/api/mes/process-flow/instances/by-business',
    method: 'get',
    params: { businessType, businessKey },
  })
}