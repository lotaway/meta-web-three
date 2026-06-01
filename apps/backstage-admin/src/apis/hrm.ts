import http from '@/utils/http'
import type { CommonResult } from '@/types/common'

// Employee Types
export interface Employee {
  id?: number
  employeeNo: string
  name: string
  gender: number
  birthday: string
  idCard: string
  mobile: string
  email: string
  departmentId: number
  departmentName: string
  positionId: number
  positionName: string
  employeeStatus: number
  employeeType: number
  entryDate: string
  contractStartDate: string
  contractEndDate: string
  baseSalary: number
  bankName: string
  bankAccount: string
  emergencyContact: string
  emergencyPhone: string
  address: string
  remark: string
  createdBy?: number
  createdAt?: string
  updatedAt?: string
}

export interface EmployeeCreateCommand {
  employeeNo: string
  name: string
  gender: number
  birthday: string
  idCard: string
  mobile: string
  email: string
  departmentId: number
  departmentName: string
  positionId: number
  positionName: string
  employeeStatus: number
  employeeType: number
  entryDate: string
  contractStartDate: string
  contractEndDate: string
  baseSalary: number
  bankName: string
  bankAccount: string
  emergencyContact: string
  emergencyPhone: string
  address: string
  remark: string
}

export interface EmployeeUpdateCommand {
  id: number
  name: string
  gender: number
  birthday: string
  idCard: string
  mobile: string
  email: string
  departmentId: number
  departmentName: string
  positionId: number
  positionName: string
  employeeStatus: number
  employeeType: number
  entryDate: string
  contractStartDate: string
  contractEndDate: string
  baseSalary: number
  bankName: string
  bankAccount: string
  emergencyContact: string
  emergencyPhone: string
  address: string
  remark: string
}

// Department Types
export interface Department {
  id?: number
  departmentCode: string
  departmentName: string
  parentId: number | null
  parentName: string | null
  level: number
  leaderId: number | null
  leaderName: string | null
  phone: string
  email: string
  status: number
  sort: number
  remark: string
  createdBy?: number
  createdAt?: string
  updatedAt?: string
  children?: Department[]
}

export interface DepartmentCreateCommand {
  departmentCode: string
  departmentName: string
  parentId: number | null
  level: number
  leaderId: number | null
  leaderName: string | null
  phone: string
  email: string
  status: number
  sort: number
  remark: string
}

export interface DepartmentUpdateCommand {
  id: number
  departmentName: string
  parentId: number | null
  level: number
  leaderId: number | null
  leaderName: string | null
  phone: string
  email: string
  status: number
  sort: number
  remark: string
}

// Employee APIs
export const getEmployeeById = (id: number) => {
  return http<CommonResult<Employee>>({ url: `/api/hrm/employee/${id}`, method: 'get' })
}

export const getAllEmployees = () => {
  return http<CommonResult<Employee[]>>({ url: '/api/hrm/employee/list', method: 'get' })
}

export const getEmployeesByDepartment = (departmentId: number) => {
  return http<CommonResult<Employee[]>>({ url: `/api/hrm/employee/department/${departmentId}`, method: 'get' })
}

export const getEmployeesByPosition = (positionId: number) => {
  return http<CommonResult<Employee[]>>({ url: `/api/hrm/employee/position/${positionId}`, method: 'get' })
}

export const getEmployeesByStatus = (status: number) => {
  return http<CommonResult<Employee[]>>({ url: `/api/hrm/employee/status/${status}`, method: 'get' })
}

export const getEmployeeByNo = (employeeNo: string) => {
  return http<CommonResult<Employee>>({ url: `/api/hrm/employee/no/${employeeNo}`, method: 'get' })
}

export const searchEmployees = (keywords: string) => {
  return http<CommonResult<Employee[]>>({ url: '/api/hrm/employee/search', method: 'get', params: { keywords } })
}

export const getFormalEmployees = () => {
  return http<CommonResult<Employee[]>>({ url: '/api/hrm/employee/formal', method: 'get' })
}

export const getProbationEmployees = () => {
  return http<CommonResult<Employee[]>>({ url: '/api/hrm/employee/probation', method: 'get' })
}

export const createEmployee = (data: EmployeeCreateCommand) => {
  return http<CommonResult<Employee>>({ url: '/api/hrm/employee', method: 'post', data })
}

export const updateEmployee = (data: EmployeeUpdateCommand) => {
  return http<CommonResult<Employee>>({ url: '/api/hrm/employee', method: 'put', data })
}

export const deleteEmployee = (id: number) => {
  return http<CommonResult<void>>({ url: `/api/hrm/employee/${id}`, method: 'delete' })
}

// Department APIs
export const getDepartmentById = (id: number) => {
  return http<CommonResult<Department>>({ url: `/api/hrm/department/${id}`, method: 'get' })
}

export const getAllDepartments = () => {
  return http<CommonResult<Department[]>>({ url: '/api/hrm/department/list', method: 'get' })
}

export const getDepartmentTree = () => {
  return http<CommonResult<Department[]>>({ url: '/api/hrm/department/tree', method: 'get' })
}

export const getDepartmentChildren = (parentId: number) => {
  return http<CommonResult<Department[]>>({ url: `/api/hrm/department/children/${parentId}`, method: 'get' })
}

export const getDepartmentsByLevel = (level: number) => {
  return http<CommonResult<Department[]>>({ url: `/api/hrm/department/level/${level}`, method: 'get' })
}

export const getDepartmentByCode = (code: string) => {
  return http<CommonResult<Department>>({ url: `/api/hrm/department/code/${code}`, method: 'get' })
}

export const createDepartment = (data: DepartmentCreateCommand) => {
  return http<CommonResult<Department>>({ url: '/api/hrm/department', method: 'post', data })
}

export const updateDepartment = (data: DepartmentUpdateCommand) => {
  return http<CommonResult<Department>>({ url: '/api/hrm/department', method: 'put', data })
}

export const deleteDepartment = (id: number) => {
  return http<CommonResult<void>>({ url: `/api/hrm/department/${id}`, method: 'delete' })
}