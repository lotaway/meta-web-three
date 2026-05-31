import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

export interface Task {
  id: number
  projectId: number
  taskCode: string
  taskName: string
  description: string
  status: string
  parentId: number
  level: number
  sort: number
  assigneeId: number
  assigneeName: string
  plannedStartDate: string
  plannedEndDate: string
  actualStartDate: string
  actualEndDate: string
  progress: number
  estimatedHours: number
  actualHours: number
  createdBy: number
  creatorName: string
  createdAt: string
  updatedAt: string
  remark: string
}

export interface TaskCreateCommand {
  projectId: number
  taskName: string
  description: string
  parentId: number
  level: number
  sort: number
  assigneeId: number
  assigneeName: string
  plannedStartDate: string
  plannedEndDate: string
  estimatedHours: number
  remark: string
}

export interface TaskUpdateCommand {
  id: number
  taskName: string
  description: string
  assigneeId: number
  assigneeName: string
  plannedStartDate: string
  plannedEndDate: string
  estimatedHours: number
  remark: string
}

export const getTaskList = (params: {
  pageNum: number
  pageSize: number
  projectId?: number
  status?: string
  assigneeId?: number
}) => {
  return http.get<CommonResult<{ list: Task[]; total: number }>>('/project-service/tasks/page', { params })
}

export const getTaskById = (id: number) => {
  return http.get<CommonResult<Task>>(`/project-service/tasks/${id}`)
}

export const getTasksByProjectId = (projectId: number) => {
  return http.get<CommonResult<Task[]>>(`/project-service/tasks/project/${projectId}`)
}

export const createTask = (data: TaskCreateCommand) => {
  return http.post<CommonResult<Task>>('/project-service/tasks', data)
}

export const updateTask = (data: TaskUpdateCommand) => {
  return http.put<CommonResult<Task>>('/project-service/tasks', data)
}

export const deleteTask = (id: number) => {
  return http.delete<CommonResult<void>>(`/project-service/tasks/${id}`)
}

export const updateTaskStatus = (id: number, status: string) => {
  return http.put<CommonResult<Task>>(`/project-service/tasks/${id}/status`, null, { params: { status } })
}

export const updateTaskProgress = (id: number, progress: number) => {
  return http.put<CommonResult<Task>>(`/project-service/tasks/${id}/progress`, null, { params: { progress } })
}