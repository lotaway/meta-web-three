import http from '@/utils/http'

export type TraceType = 'PRODUCT' | 'BATCH' | 'MATERIAL' | 'SN' | 'WORK_ORDER' | 'PROCESS' | 'QC' | 'EQUIPMENT' | 'OPERATOR'
export type TraceSource = 'WORK_ORDER' | 'PRODUCTION_TASK' | 'MATERIAL_ISSUE' | 'QC_INSPECTION' | 'EQUIPMENT' | 'ANDON'
export type RelationType = 'CONSUMED' | 'SPLIT_FROM' | 'PARENT' | 'BIND'

export interface TraceRelation {
  relatedCode: string
  relatedType: TraceType
  relationType: string
  quantity?: number
}

export interface TraceRecord {
  id?: number
  traceCode: string
  traceType: TraceType
  productCode?: string
  batchNo?: string
  sn?: string
  sourceTraceCode?: string
  source?: TraceSource
  relations: TraceRelation[]
  createdAt?: string
}

export interface TraceChain {
  root: TraceRecord
  forwardPath: TraceRecord[]
  backwardPath: TraceRecord[]
}

export interface TraceModel {
  id?: number
  modelCode: string
  modelName: string
  productType: string
  relationConfig?: {
    enableBatchTrace?: boolean
    enableSnTrace?: boolean
    enableMaterialTrace?: boolean
    enableProcessTrace?: boolean
    enableQualityTrace?: boolean
    enableEquipmentTrace?: boolean
    traceLevels?: TraceLevel[]
  }
  isEnabled?: boolean
  createdAt?: string
  updatedAt?: string
}

export interface TraceLevel {
  levelCode: string
  levelName: string
  traceType: TraceType
  parentLevelCode?: string
  isRequired?: boolean
}

// ========== Trace Records ==========

export function getTraceRecordListAPI(params?: {
  productCode?: string
  batchNo?: string
  workOrderNo?: string
  sn?: string
}) {
  return http<TraceRecord[]>({ url: '/api/mes/trace/records', method: 'get', params })
}

export function getTraceRecordByIdAPI(id: number) {
  return http<TraceRecord>({ url: `/api/mes/trace/records/${id}`, method: 'get' })
}

export function createTraceRecordAPI(data: {
  traceCode: string
  traceType: TraceType
  productCode?: string
  productName?: string
  batchNo?: string
  sn?: string
  source?: TraceSource
  workOrderNo?: string
}) {
  return http<TraceRecord>({ url: '/api/mes/trace/records', method: 'post', data })
}

export function linkMaterialAPI(id: number, data: { materialCode: string; batchNo?: string; quantity?: number }) {
  return http<TraceRecord>({ url: `/api/mes/trace/records/${id}/material`, method: 'post', data })
}

export function linkEquipmentAPI(id: number, data: { equipmentCode: string; equipmentName?: string }) {
  return http<TraceRecord>({ url: `/api/mes/trace/records/${id}/equipment`, method: 'post', data })
}

export function linkOperatorAPI(id: number, data: { operatorCode: string; operatorName?: string }) {
  return http<TraceRecord>({ url: `/api/mes/trace/records/${id}/operator`, method: 'post', data })
}

export function linkQcAPI(id: number, data: { qcRecordCode: string }) {
  return http<TraceRecord>({ url: `/api/mes/trace/records/${id}/qc`, method: 'post', data })
}

export function forwardTraceAPI(traceCode: string) {
  return http<TraceRecord[]>({ url: '/api/mes/trace/records/forward', method: 'get', params: { traceCode } })
}

export function backwardTraceAPI(traceCode: string) {
  return http<TraceRecord[]>({ url: '/api/mes/trace/records/backward', method: 'get', params: { traceCode } })
}

export function getTraceChainAPI(traceCode: string) {
  return http<TraceChain>({ url: '/api/mes/trace/records/trace-chain', method: 'get', params: { traceCode } })
}

export function deleteTraceRecordAPI(id: number) {
  return http({ url: `/api/mes/trace/records/${id}`, method: 'delete' })
}

// ========== Trace Models ==========

export function getTraceModelListAPI(productType?: string) {
  return http<TraceModel[]>({ url: '/api/mes/trace/models', method: 'get', params: productType ? { productType } : {} })
}

export function getTraceModelByIdAPI(id: number) {
  return http<TraceModel>({ url: `/api/mes/trace/models/${id}`, method: 'get' })
}

export function createTraceModelAPI(data: {
  modelCode: string
  modelName: string
  productType: string
}) {
  return http<TraceModel>({ url: '/api/mes/trace/models', method: 'post', data })
}

export function updateTraceModelAPI(id: number, data: {
  modelName?: string
  productType?: string
  enableBatch?: boolean
  enableSn?: boolean
  enableMaterial?: boolean
  enableProcess?: boolean
  enableQuality?: boolean
  enableEquipment?: boolean
}) {
  return http<TraceModel>({ url: `/api/mes/trace/models/${id}`, method: 'put', data })
}

export function deleteTraceModelAPI(id: number) {
  return http({ url: `/api/mes/trace/models/${id}`, method: 'delete' })
}
