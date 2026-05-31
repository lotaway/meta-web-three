import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

export interface Project {
  id: number
  projectCode: string
  projectName: string
  description: string
  status: string
  departmentId: number
  departmentName: string
  managerId: number
  managerName: string
  startDate: string
  endDate: string
  budgetAmount: number
  usedAmount: number
  currency: string
  progress: number
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  remark: string
}

export interface ProjectCreateCommand {
  projectCode: string
  projectName: string
  description: string
  departmentId: number
  departmentName: string
  managerId: number
  managerName: string
  startDate: string
  endDate: string
  budgetAmount: number
  currency: string
  remark: string
}

export interface ProjectUpdateCommand {
  id: number
  projectName: string
  description: string
  departmentId: number
  departmentName: string
  managerId: number
  managerName: string
  startDate: string
  endDate: string
  budgetAmount: number
  remark: string
}

export const getProjectList = (params: {
  pageNum: number
  pageSize: number
  keyword?: string
  status?: string
  departmentId?: number
}) => {
  return http.get<CommonResult<{ list: Project[]; total: number }>>('/project-service/projects/page', { params })
}

export const getProjectById = (id: number) => {
  return http.get<CommonResult<Project>>(`/project-service/projects/${id}`)
}

export const createProject = (data: ProjectCreateCommand) => {
  return http.post<CommonResult<Project>>('/project-service/projects', data)
}

export const updateProject = (data: ProjectUpdateCommand) => {
  return http.put<CommonResult<Project>>('/project-service/projects', data)
}

export const deleteProject = (id: number) => {
  return http.delete<CommonResult<void>>(`/project-service/projects/${id}`)
}

export const updateProjectStatus = (id: number, status: string) => {
  return http.put<CommonResult<Project>>(`/project-service/projects/${id}/status`, null, { params: { status } })
}

export const updateProjectProgress = (id: number, progress: number) => {
  return http.put<CommonResult<Project>>(`/project-service/projects/${id}/progress`, null, { params: { progress } })
}