import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

export interface CostRecord {
  id: number
  projectId: number
  projectName: string
  costType: string
  costCode: string
  costName: string
  amount: number
  currency: string
  costDate: string
  description: string
  status: string
  departmentId: number
  departmentName: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  remark: string
}

export interface CostRecordCreateCommand {
  projectId: number
  projectName: string
  costType: string
  costCode: string
  costName: string
  amount: number
  currency: string
  costDate: string
  description: string
  departmentId: number
  departmentName: string
  remark: string
}

export interface CostRecordUpdateCommand {
  id: number
  projectId: number
  projectName: string
  costType: string
  costCode: string
  costName: string
  amount: number
  currency: string
  costDate: string
  description: string
  departmentId: number
  departmentName: string
  remark: string
}

export const getCostRecordList = (params: {
  pageNum: number
  pageSize: number
  projectId?: number
  costType?: string
  status?: string
  startDate?: string
  endDate?: string
}) => {
  return http.get<CommonResult<{ list: CostRecord[]; total: number }>>('/project-service/cost-records/page', { params })
}

export const getCostRecordById = (id: number) => {
  return http.get<CommonResult<CostRecord>>(`/project-service/cost-records/${id}`)
}

export const getCostRecordsByProjectId = (projectId: number) => {
  return http.get<CommonResult<CostRecord[]>>(`/project-service/cost-records/project/${projectId}`)
}

export const createCostRecord = (data: CostRecordCreateCommand) => {
  return http.post<CommonResult<CostRecord>>('/project-service/cost-records', data)
}

export const updateCostRecord = (data: CostRecordUpdateCommand) => {
  return http.put<CommonResult<CostRecord>>('/project-service/cost-records', data)
}

export const deleteCostRecord = (id: number) => {
  return http.delete<CommonResult<void>>(`/project-service/cost-records/${id}`)
}