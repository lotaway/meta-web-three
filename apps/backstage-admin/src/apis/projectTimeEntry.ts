import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

export interface TimeEntry {
  id: number
  projectId: number
  projectName: string
  taskId: number
  taskName: string
  employeeId: number
  employeeName: string
  workDate: string
  hours: number
  workType: string
  description: string
  status: string
  approverId: number
  approverName: string
  approvedAt: string
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  remark: string
}

export interface TimeEntryCreateCommand {
  projectId: number
  projectName: string
  taskId: number
  taskName: string
  employeeId: number
  employeeName: string
  workDate: string
  hours: number
  workType: string
  description: string
  remark: string
}

export interface TimeEntryUpdateCommand {
  id: number
  projectId: number
  projectName: string
  taskId: number
  taskName: string
  workDate: string
  hours: number
  workType: string
  description: string
  remark: string
}

export const getTimeEntryList = (params: {
  pageNum: number
  pageSize: number
  projectId?: number
  employeeId?: number
  status?: string
  startDate?: string
  endDate?: string
}) => {
  return http.get<CommonResult<{ list: TimeEntry[]; total: number }>>('/project-service/time-entries/page', { params })
}

export const getTimeEntryById = (id: number) => {
  return http.get<CommonResult<TimeEntry>>(`/project-service/time-entries/${id}`)
}

export const getTimeEntriesByProjectId = (projectId: number) => {
  return http.get<CommonResult<TimeEntry[]>>(`/project-service/time-entries/project/${projectId}`)
}

export const getTimeEntriesByEmployeeId = (employeeId: number) => {
  return http.get<CommonResult<TimeEntry[]>>(`/project-service/time-entries/employee/${employeeId}`)
}

export const createTimeEntry = (data: TimeEntryCreateCommand) => {
  return http.post<CommonResult<TimeEntry>>('/project-service/time-entries', data)
}

export const updateTimeEntry = (data: TimeEntryUpdateCommand) => {
  return http.put<CommonResult<TimeEntry>>('/project-service/time-entries', data)
}

export const deleteTimeEntry = (id: number) => {
  return http.delete<CommonResult<void>>(`/project-service/time-entries/${id}`)
}

export const approveTimeEntry = (id: number, approverId: number, approverName: string) => {
  return http.put<CommonResult<TimeEntry>>(`/project-service/time-entries/${id}/approve`, null, { params: { approverId, approverName } })
}

export const rejectTimeEntry = (id: number) => {
  return http.put<CommonResult<TimeEntry>>(`/project-service/time-entries/${id}/reject`, null)
}