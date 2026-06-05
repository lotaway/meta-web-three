import http from '@/utils/http'

export type OperatorStatus = 'ACTIVE' | 'INACTIVE' | 'ON_LEAVE' | 'TERMINATED'
export type SkillLevel = 'TRAINEE' | 'JUNIOR' | 'MIDDLE' | 'SENIOR' | 'MASTER'
export type AttendanceStatus = 'PRESENT' | 'LATE' | 'ABSENT' | 'HALF_DAY' | 'OVERTIME' | 'VACATION' | 'SICK' | 'BUSINESS_TRIP'
export type RecordType = 'REGULAR' | 'OVERTIME' | 'VACATION' | 'SICK'
export type RecordStatus = 'DRAFT' | 'SUBMITTED' | 'APPROVED' | 'REJECTED'
export type ShiftType = 'DAY' | 'NIGHT' | 'MIDDLE' | 'ROTATING'
export type AssignmentStatus = 'ACTIVE' | 'INACTIVE'

export interface OperatorSkill {
  id?: number
  operatorId?: number
  skillCode: string
  skillName: string
  skillLevel: SkillLevel
  certified?: boolean
  certifiedAt?: string
  expiryAt?: string
}

export interface Operator {
  id?: number
  operatorCode: string
  operatorName: string
  department: string
  jobTitle?: string
  shiftGroup: string
  status: OperatorStatus
  phone?: string
  email?: string
  idCardNo?: string
  hireDate?: string
  skills?: OperatorSkill[]
  remark?: string
  createdBy?: string
  createdAt?: string
  updatedAt?: string
}

export interface Attendance {
  id?: number
  operatorId: number
  operatorCode: string
  operatorName: string
  attendanceDate: string
  clockIn?: string
  clockOut?: string
  scheduledStart: string
  scheduledEnd: string
  status: AttendanceStatus
  overtime?: boolean
  remark?: string
}

export interface TimeRecord {
  id?: number
  operatorId: number
  operatorCode: string
  operatorName: string
  workOrderNo?: string
  taskNo?: string
  operationCode?: string
  workCenterId?: string
  recordDate: string
  startTime: string
  endTime?: string
  totalHours?: number
  recordType: RecordType
  status: RecordStatus
  approvedBy?: string
  approvedAt?: string
  remark?: string
}

export interface WorkCenterAssignment {
  id?: number
  operatorId: number
  workCenterId: string
  workCenterName: string
  startDate?: string
  endDate?: string
  shiftType: ShiftType
  status: AssignmentStatus
  remark?: string
}

// ========== Operator APIs ==========
export function getOperatorListAPI(params?: { department?: string; status?: OperatorStatus }) {
  return http<Operator[]>({ url: '/api/mes/labor/operators', method: 'get', params })
}
export function getOperatorByIdAPI(id: number) {
  return http<Operator>({ url: `/api/mes/labor/operators/${id}`, method: 'get' })
}
export function createOperatorAPI(data: { operatorCode: string; operatorName: string; department: string; shiftGroup: string }) {
  return http<Operator>({ url: '/api/mes/labor/operators', method: 'post', data })
}
export function updateOperatorAPI(id: number, data: { operatorName?: string; department?: string; jobTitle?: string; shiftGroup?: string; phone?: string; email?: string }) {
  return http<Operator>({ url: `/api/mes/labor/operators/${id}`, method: 'put', data })
}
export function changeOperatorStatusAPI(id: number, status: OperatorStatus) {
  return http<Operator>({ url: `/api/mes/labor/operators/${id}/status`, method: 'put', params: { status } })
}
export function addSkillAPI(id: number, data: { skillCode: string; skillName: string; skillLevel: SkillLevel }) {
  return http<Operator>({ url: `/api/mes/labor/operators/${id}/skills`, method: 'post', data })
}
export function deleteOperatorAPI(id: number) {
  return http({ url: `/api/mes/labor/operators/${id}`, method: 'delete' })
}

// ========== Attendance APIs ==========
export function clockInAPI(operatorId: number) {
  return http<Attendance>({ url: '/api/mes/labor/attendance/clock-in', method: 'post', params: { operatorId } })
}
export function clockOutAPI(operatorId: number) {
  return http<Attendance>({ url: '/api/mes/labor/attendance/clock-out', method: 'post', params: { operatorId } })
}
export function getAttendanceListAPI(params?: { operatorId?: number; date?: string }) {
  return http<Attendance[]>({ url: '/api/mes/labor/attendance', method: 'get', params })
}

// ========== Time Record APIs ==========
export function startTimeRecordAPI(operatorId: number, recordType?: RecordType) {
  return http<TimeRecord>({ url: '/api/mes/labor/time-records/start', method: 'post', params: { operatorId, recordType } })
}
export function endTimeRecordAPI(id: number) {
  return http<TimeRecord>({ url: `/api/mes/labor/time-records/${id}/end`, method: 'post' })
}
export function submitTimeRecordAPI(id: number) {
  return http<TimeRecord>({ url: `/api/mes/labor/time-records/${id}/submit`, method: 'post' })
}
export function approveTimeRecordAPI(id: number, approvedBy: string) {
  return http<TimeRecord>({ url: `/api/mes/labor/time-records/${id}/approve`, method: 'post', params: { approvedBy } })
}
export function rejectTimeRecordAPI(id: number, approvedBy: string) {
  return http<TimeRecord>({ url: `/api/mes/labor/time-records/${id}/reject`, method: 'post', params: { approvedBy } })
}
export function getTimeRecordListAPI(params?: { operatorId?: number; status?: RecordStatus; startDate?: string; endDate?: string }) {
  return http<TimeRecord[]>({ url: '/api/mes/labor/time-records', method: 'get', params })
}
export function deleteTimeRecordAPI(id: number) {
  return http({ url: `/api/mes/labor/time-records/${id}`, method: 'delete' })
}

// ========== Assignment APIs ==========
export function createAssignmentAPI(data: { operatorId: number; workCenterId: string; workCenterName: string; shiftType: ShiftType }) {
  return http<WorkCenterAssignment>({ url: '/api/mes/labor/assignments', method: 'post', data })
}
export function endAssignmentAPI(id: number) {
  return http({ url: `/api/mes/labor/assignments/${id}/end`, method: 'post' })
}
export function getAssignmentListAPI(params?: { operatorId?: number; workCenterId?: string }) {
  return http<WorkCenterAssignment[]>({ url: '/api/mes/labor/assignments', method: 'get', params })
}
