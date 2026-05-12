import type { CommonPage, PageParam } from '@/types/common'
import type { UmsMenu } from '@/types/menu'
import type { UmsRole } from '@/types/role'
import http from '@/utils/http'
export function getRoleListAllAPI() {
  return http<UmsRole[]>({
    url: '/role/listAll',
    method: 'get',
  })
}
export function getRoleListAPI(params: PageParam) {
  return http<CommonPage<UmsRole>>({
    url: '/role/list',
    method: 'get',
    params: params,
  })
}
export function roleCreateAPI(data: UmsRole) {
  return http({
    url: '/role/create',
    method: 'post',
    data: data,
  })
}
export function roleUpdateByIdAPI(id: number, data: UmsRole) {
  return http({
    url: '/role/update/' + id,
    method: 'post',
    data: data,
  })
}
export function roleUpdateStatusAPI(id: number, params: { status: number }) {
  return http({
    url: '/role/updateStatus/' + id,
    method: 'post',
    params: params,
  })
}
export function roleDeleteByIdsAPI(params: { ids: string }) {
  return http({
    url: '/role/delete',
    method: 'post',
    params: params,
  })
}
export function roleListMenuByRoleIdAPI(id: number) {
  return http<UmsMenu[]>({
    url: '/role/listMenu/' + id,
    method: 'get',
  })
}
export function roleAllocMenuAPI(params: {
  /** 角色ID */
  roleId: number
  /** 菜单ID，多个以逗号分割 */
  menuIds: string
}) {
  return http({
    url: '/role/allocMenu',
    method: 'post',
    params: params,
  })
}
export function roleListResourceById(id: number) {
  return http({
    url: '/role/listResource/' + id,
    method: 'get',
  })
}
export function roleAllocResourceAPI(params: {
  /** 角色ID */
  roleId: number
  /** 资源ID，多个以逗号分割 */
  resourceIds: string
}) {
  return http({
    url: '/role/allocResource',
    method: 'post',
    params: params,
  })
}
