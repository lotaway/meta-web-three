import http from '@/utils/http'

// ==================== Inspection Type ====================
export type InspectionCategory = 'INCOMING' | 'PROCESS' | 'FINAL' | 'OUTGOING' | 'CUSTOM'

export interface QcInspectionType {
  id?: number
  typeCode: string
  typeName: string
  category: InspectionCategory
  description?: string
  applicableProducts?: string
  defaultSamplingPlan?: string
  defaultAql?: string
  defaultTimeoutHours?: number
  requireCertificate?: boolean
  requireTestReport?: boolean
  status: 'ACTIVE' | 'INACTIVE'
  sortOrder?: number
  createdAt?: string
  updatedAt?: string
}

export const getInspectionTypeListAPI = () => {
  return http<QcInspectionType[]>({ method: 'get', url: '/api/mes/qc/inspection-type' })
}

export const getInspectionTypeByIdAPI = (id: number) => {
  return http<QcInspectionType>({ method: 'get', url: `/api/mes/qc/inspection-type/${id}` })
}

export const getInspectionTypeByCategoryAPI = (category: string) => {
  return http<QcInspectionType[]>({ method: 'get', url: `/api/mes/qc/inspection-type/category/${category}` })
}

export const createInspectionTypeAPI = (data: Partial<QcInspectionType>) => {
  return http<QcInspectionType>({ method: 'post', url: '/api/mes/qc/inspection-type', data })
}

export const updateInspectionTypeAPI = (id: number, data: Partial<QcInspectionType>) => {
  return http<QcInspectionType>({ method: 'put', url: `/api/mes/qc/inspection-type/${id}`, data })
}

export const deleteInspectionTypeAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/inspection-type/${id}` })
}

export const activateInspectionTypeAPI = (id: number) => {
  return http<QcInspectionType>({ method: 'post', url: `/api/mes/qc/inspection-type/${id}/activate` })
}

export const deactivateInspectionTypeAPI = (id: number) => {
  return http<QcInspectionType>({ method: 'post', url: `/api/mes/qc/inspection-type/${id}/deactivate` })
}

// ==================== Inspection Plan ====================
export type PlanStatus = 'DRAFT' | 'ACTIVE' | 'SUSPENDED' | 'ARCHIVED'

export interface QcPlanItem {
  id?: number
  planId: number
  itemId: number
  inspectionItem?: QcInspectionItem
  sequence: number
  isMandatory: boolean
  samplingMethod?: string
  aql?: string
  acceptanceNumber?: number
  rejectionNumber?: number
}

export interface QcInspectionPlan {
  id?: number
  planCode: string
  planName: string
  inspectionType: string
  applicableProducts?: string
  status: PlanStatus
  effectiveDate?: string
  expiryDate?: string
  version?: number
  description?: string
  planItems?: QcPlanItem[]
  createdAt?: string
  updatedAt?: string
}

export interface QcInspectionItem {
  id?: number
  itemCode: string
  itemName: string
  inspectionMethod?: string
  equipmentRequired?: string
  standardValue?: string
  upperLimit?: number
  lowerLimit?: number
  unit?: string
  status: 'ACTIVE' | 'INACTIVE'
  createdAt?: string
  updatedAt?: string
}

export const getInspectionPlanListAPI = () => {
  return http<QcInspectionPlan[]>({ method: 'get', url: '/api/mes/qc/inspection-plan' })
}

export const getInspectionPlanByIdAPI = (id: number) => {
  return http<QcInspectionPlan>({ method: 'get', url: `/api/mes/qc/inspection-plan/${id}` })
}

export const getInspectionPlanByTypeAPI = (inspectionType: string) => {
  return http<QcInspectionPlan[]>({ method: 'get', url: `/api/mes/qc/inspection-plan/type/${inspectionType}` })
}

export const getInspectionPlanByStatusAPI = (status: string) => {
  return http<QcInspectionPlan[]>({ method: 'get', url: `/api/mes/qc/inspection-plan/status/${status}` })
}

export const createInspectionPlanAPI = (data: Partial<QcInspectionPlan>) => {
  return http<QcInspectionPlan>({ method: 'post', url: '/api/mes/qc/inspection-plan', data })
}

export const updateInspectionPlanAPI = (id: number, data: Partial<QcInspectionPlan>) => {
  return http<QcInspectionPlan>({ method: 'put', url: `/api/mes/qc/inspection-plan/${id}`, data })
}

export const deleteInspectionPlanAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/inspection-plan/${id}` })
}

export const activateInspectionPlanAPI = (id: number) => {
  return http<QcInspectionPlan>({ method: 'post', url: `/api/mes/qc/inspection-plan/${id}/activate` })
}

export const deactivateInspectionPlanAPI = (id: number) => {
  return http<QcInspectionPlan>({ method: 'post', url: `/api/mes/qc/inspection-plan/${id}/deactivate` })
}

export const addInspectionPlanItemAPI = (planId: number, data: { itemId: number; itemSequence: number }) => {
  return http<QcPlanItem>({ method: 'post', url: `/api/mes/qc/inspection-plan/${planId}/items`, data })
}

export const removeInspectionPlanItemAPI = (planId: number, itemId: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/inspection-plan/${planId}/items/${itemId}` })
}

// ==================== Inspection Item ====================
export const getInspectionItemListAPI = () => {
  return http<QcInspectionItem[]>({ method: 'get', url: '/api/mes/qc/inspection-item' })
}

export const getInspectionItemByIdAPI = (id: number) => {
  return http<QcInspectionItem>({ method: 'get', url: `/api/mes/qc/inspection-item/${id}` })
}

export const getInspectionItemByCategoryAPI = (itemCategory: string) => {
  return http<QcInspectionItem[]>({ method: 'get', url: `/api/mes/qc/inspection-item/category/${itemCategory}` })
}

export const getInspectionItemByStatusAPI = (status: string) => {
  return http<QcInspectionItem[]>({ method: 'get', url: `/api/mes/qc/inspection-item/status/${status}` })
}

export const createInspectionItemAPI = (data: Partial<QcInspectionItem>) => {
  return http<QcInspectionItem>({ method: 'post', url: '/api/mes/qc/inspection-item', data })
}

export const updateInspectionItemAPI = (id: number, data: Partial<QcInspectionItem>) => {
  return http<QcInspectionItem>({ method: 'put', url: `/api/mes/qc/inspection-item/${id}`, data })
}

export const deleteInspectionItemAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/inspection-item/${id}` })
}

export const activateInspectionItemAPI = (id: number) => {
  return http<QcInspectionItem>({ method: 'post', url: `/api/mes/qc/inspection-item/${id}/activate` })
}

export const deactivateInspectionItemAPI = (id: number) => {
  return http<QcInspectionItem>({ method: 'post', url: `/api/mes/qc/inspection-item/${id}/deactivate` })
}

// ==================== Defect Code ====================
export type DefectCategory = '外观' | '尺寸' | '功能' | '性能' | '材料' | '装配' | '其他'
export type DefectSeverity = 'CRITICAL' | 'MAJOR' | 'MINOR' | 'OBSERVATION'

export interface DefectCode {
  id?: number
  defectCode: string
  defectName: string
  category: DefectCategory
  severity: DefectSeverity
  description?: string
  dispositionGuide?: string
  isEnabled?: boolean
  sortOrder?: number
  createdAt?: string
  updatedAt?: string
}

export const getDefectCodeListAPI = () => {
  return http<DefectCode[]>({ method: 'get', url: '/api/mes/qc/defect-code' })
}

export const getDefectCodeByIdAPI = (id: number) => {
  return http<DefectCode>({ method: 'get', url: `/api/mes/qc/defect-code/${id}` })
}

export const getDefectCodeByCodeAPI = (code: string) => {
  return http<DefectCode>({ method: 'get', url: `/api/mes/qc/defect-code/code/${code}` })
}

export const getDefectCodeByCategoryAPI = (category: string) => {
  return http<DefectCode[]>({ method: 'get', url: `/api/mes/qc/defect-code/category/${category}` })
}

export const getDefectCodeBySeverityAPI = (severity: string) => {
  return http<DefectCode[]>({ method: 'get', url: `/api/mes/qc/defect-code/severity/${severity}` })
}

export const getEnabledDefectCodeListAPI = () => {
  return http<DefectCode[]>({ method: 'get', url: '/api/mes/qc/defect-code/enabled' })
}

export const createDefectCodeAPI = (data: Partial<DefectCode>) => {
  return http<DefectCode>({ method: 'post', url: '/api/mes/qc/defect-code', data })
}

export const updateDefectCodeAPI = (id: number, data: Partial<DefectCode>) => {
  return http<DefectCode>({ method: 'put', url: `/api/mes/qc/defect-code/${id}`, data })
}

export const deleteDefectCodeAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/defect-code/${id}` })
}

export const enableDefectCodeAPI = (id: number) => {
  return http<DefectCode>({ method: 'post', url: `/api/mes/qc/defect-code/${id}/enable` })
}

export const disableDefectCodeAPI = (id: number) => {
  return http<DefectCode>({ method: 'post', url: `/api/mes/qc/defect-code/${id}/disable` })
}

// ==================== Trigger Rule ====================
export type TriggerType = 'AUTO' | 'MANUAL' | 'SCHEDULED' | 'EVENT_BASED'
export type RuleStatus = 'ACTIVE' | 'INACTIVE'

export interface QcTriggerRule {
  id?: number
  ruleCode: string
  ruleName: string
  triggerType: TriggerType
  targetEntity: string
  conditionExpression?: string
  actionType?: string
  actionConfig?: string
  priority?: number
  status: RuleStatus
  description?: string
  createdAt?: string
  updatedAt?: string
}

export const getTriggerRuleListAPI = () => {
  return http<QcTriggerRule[]>({ method: 'get', url: '/api/mes/qc/trigger-rule' })
}

export const getTriggerRuleByIdAPI = (id: number) => {
  return http<QcTriggerRule>({ method: 'get', url: `/api/mes/qc/trigger-rule/${id}` })
}

export const getTriggerRuleByCodeAPI = (code: string) => {
  return http<QcTriggerRule>({ method: 'get', url: `/api/mes/qc/trigger-rule/code/${code}` })
}

export const getTriggerRuleByTypeAPI = (triggerType: string) => {
  return http<QcTriggerRule[]>({ method: 'get', url: `/api/mes/qc/trigger-rule/type/${triggerType}` })
}

export const getEnabledTriggerRuleListAPI = () => {
  return http<QcTriggerRule[]>({ method: 'get', url: '/api/mes/qc/trigger-rule/enabled' })
}

export const createTriggerRuleAPI = (data: Partial<QcTriggerRule>) => {
  return http<QcTriggerRule>({ method: 'post', url: '/api/mes/qc/trigger-rule', data })
}

export const updateTriggerRuleAPI = (id: number, data: Partial<QcTriggerRule>) => {
  return http<QcTriggerRule>({ method: 'put', url: `/api/mes/qc/trigger-rule/${id}`, data })
}

export const deleteTriggerRuleAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/trigger-rule/${id}` })
}

export const enableTriggerRuleAPI = (id: number) => {
  return http<QcTriggerRule>({ method: 'post', url: `/api/mes/qc/trigger-rule/${id}/enable` })
}

export const disableTriggerRuleAPI = (id: number) => {
  return http<QcTriggerRule>({ method: 'post', url: `/api/mes/qc/trigger-rule/${id}/disable` })
}

// ==================== SPC Control Chart ====================
export type ChartType = 'XBAR_R' | 'XBAR_S' | 'X_MR' | 'P_CHART' | 'NP_CHART' | 'C_CHART' | 'U_CHART'

export interface SpcDataPoint {
  subgroupId: number
  sampleValue: number
  timestamp?: string
}

export interface SpcControlChart {
  id?: number
  chartCode: string
  chartName: string
  chartType: ChartType
  parameterName: string
  unit?: string
  targetValue?: number
  usl?: number
  lsl?: number
  ucl?: number
  lcl?: number
  centerLine?: number
  status: 'ACTIVE' | 'INACTIVE'
  description?: string
  createdAt?: string
  updatedAt?: string
}

export const getSpcControlChartListAPI = () => {
  return http<SpcControlChart[]>({ method: 'get', url: '/api/mes/qc/spc-control-chart' })
}

export const getSpcControlChartByIdAPI = (id: number) => {
  return http<SpcControlChart>({ method: 'get', url: `/api/mes/qc/spc-control-chart/${id}` })
}

export const getSpcControlChartByCodeAPI = (code: string) => {
  return http<SpcControlChart>({ method: 'get', url: `/api/mes/qc/spc-control-chart/code/${code}` })
}

export const getSpcControlChartByTypeAPI = (type: string) => {
  return http<SpcControlChart[]>({ method: 'get', url: `/api/mes/qc/spc-control-chart/type/${type}` })
}

export const getEnabledSpcControlChartListAPI = () => {
  return http<SpcControlChart[]>({ method: 'get', url: '/api/mes/qc/spc-control-chart/enabled' })
}

export const createSpcControlChartAPI = (data: Partial<SpcControlChart>) => {
  return http<SpcControlChart>({ method: 'post', url: '/api/mes/qc/spc-control-chart', data })
}

export const updateSpcControlChartAPI = (id: number, data: Partial<SpcControlChart>) => {
  return http<SpcControlChart>({ method: 'put', url: `/api/mes/qc/spc-control-chart/${id}`, data })
}

export const deleteSpcControlChartAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/spc-control-chart/${id}` })
}

export const enableSpcControlChartAPI = (id: number) => {
  return http<SpcControlChart>({ method: 'post', url: `/api/mes/qc/spc-control-chart/${id}/enable` })
}

export const disableSpcControlChartAPI = (id: number) => {
  return http<SpcControlChart>({ method: 'post', url: `/api/mes/qc/spc-control-chart/${id}/disable` })
}

export const calculateSpcControlLimitsAPI = (chartId: number, dataPoints: SpcDataPoint[]) => {
  return http<{ ucl: number; lcl: number; centerLine: number }>({
    method: 'post',
    url: `/api/mes/qc/spc-control-chart/${chartId}/calculate`,
    data: { dataPoints }
  })
}

// ==================== NonConformance Disposition ====================
export type DispositionType = 'SCRAP' | 'REWORK' | 'RETURN' | 'USE_AS_IS' | '降级使用'

export interface DispositionStep {
  stepOrder: number
  stepName: string
  action: string
  assigneeRole?: string
  requiresApproval?: boolean
  timeoutHours?: number
}

export interface NonConformanceDisposition {
  id?: number
  dispositionCode: string
  dispositionName: string
  type: DispositionType
  steps?: DispositionStep[]
  isEnabled?: boolean
  sortOrder?: number
  createdAt?: string
  updatedAt?: string
}

export const getNonConformanceDispositionListAPI = () => {
  return http<NonConformanceDisposition[]>({ method: 'get', url: '/api/mes/qc/non-conformance' })
}

export const getNonConformanceDispositionByIdAPI = (id: number) => {
  return http<NonConformanceDisposition>({ method: 'get', url: `/api/mes/qc/non-conformance/${id}` })
}

export const getNonConformanceDispositionByCodeAPI = (code: string) => {
  return http<NonConformanceDisposition>({ method: 'get', url: `/api/mes/qc/non-conformance/code/${code}` })
}

export const getNonConformanceDispositionByTypeAPI = (type: string) => {
  return http<NonConformanceDisposition[]>({ method: 'get', url: `/api/mes/qc/non-conformance/type/${type}` })
}

export const getEnabledNonConformanceDispositionListAPI = () => {
  return http<NonConformanceDisposition[]>({ method: 'get', url: '/api/mes/qc/non-conformance/enabled' })
}

export const createNonConformanceDispositionAPI = (data: Partial<NonConformanceDisposition>) => {
  return http<NonConformanceDisposition>({ method: 'post', url: '/api/mes/qc/non-conformance', data })
}

export const updateNonConformanceDispositionAPI = (id: number, data: Partial<NonConformanceDisposition>) => {
  return http<NonConformanceDisposition>({ method: 'put', url: `/api/mes/qc/non-conformance/${id}`, data })
}

export const deleteNonConformanceDispositionAPI = (id: number) => {
  return http({ method: 'delete', url: `/api/mes/qc/non-conformance/${id}` })
}

export const enableNonConformanceDispositionAPI = (id: number) => {
  return http<NonConformanceDisposition>({ method: 'post', url: `/api/mes/qc/non-conformance/${id}/enable` })
}

export const disableNonConformanceDispositionAPI = (id: number) => {
  return http<NonConformanceDisposition>({ method: 'post', url: `/api/mes/qc/non-conformance/${id}/disable` })
}