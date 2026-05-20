import type { LoginParam, LoginResult, UmsAdmin, UserInfoResult } from '@/types/admin'
import type { CommonPage, PageParam } from '@/types/common'
import type { UmsRole } from '@/types/role'
import http from '@/utils/http'
export function adminLoginAPI(data: LoginParam) {
  return http<LoginResult>({
    method: 'POST',
    url: '/admin/login',
    data: data,
  })
}
export function adminLogoutAPI() {
  return http({
    method: 'POST',
    url: '/admin/logout',
  })
}
export function getAdminInfoAPI() {
  return http<UserInfoResult>({
    method: 'GET',
    url: '/admin/info',
  })
}
export function getAdminListAPI(params: PageParam) {
  return http<CommonPage<UmsAdmin>>({
    url: '/admin/list',
    method: 'get',
    params: params,
  })
}
export function adminRegisterAPI(data: UmsAdmin) {
  return http({
    url: '/admin/register',
    method: 'post',
    data: data,
  })
}
export function adminUpdateByIdAPI(id: number, data: UmsAdmin) {
  return http({
    url: '/admin/update/' + id,
    method: 'post',
    data: data,
  })
}
export function adminUpdateStatusByIdAPI(id: number, params: { status: number }) {
  return http({
    url: '/admin/updateStatus/' + id,
    method: 'post',
    params: params,
  })
}
export function adminDeleteByIdAPI(id: number) {
  return http({
    url: '/admin/delete/' + id,
    method: 'post',
  })
}
export function getRoleByAdminIdAPI(id: number) {
  return http<UmsRole[]>({
    url: '/admin/role/' + id,
    method: 'get',
  })
}
export function adminRoleUpdateAPI(params: { roleIds: string; adminId: number }) {
  return http({
    url: '/admin/role/update',
    method: 'post',
    params: params,
  })
}
