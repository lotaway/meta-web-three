import type { CommonPage, PageParam } from '@/types/common'
import type { UmsMenu, UmsMenuNode } from '@/types/menu'
import http from '@/utils/http'
export function getMenuTreeListAPI() {
  return http<UmsMenuNode[]>({
    url: '/menu/treeList',
    method: 'get',
  })
}
export function getMenuListByParentIdAPI(parentId: number, params: PageParam) {
  return http<CommonPage<UmsMenu>>({
    url: '/menu/list/' + parentId,
    method: 'get',
    params: params,
  })
}
export function deleteMenuByIdAPI(id: number) {
  return http({
    url: '/menu/delete/' + id,
    method: 'post',
  })
}
export function menuCreateAPI(data: UmsMenu) {
  return http({
    url: '/menu/create',
    method: 'post',
    data: data,
  })
}
export function updateMenu(id: number, data: UmsMenu) {
  return http({
    url: '/menu/update/' + id,
    method: 'post',
    data: data,
  })
}
export function getMenuByIdAPI(id: number) {
  return http<UmsMenu>({
    url: '/menu/' + id,
    method: 'get',
  })
}
export function menuUpdateHiddenByIdAPI(id: number, params: { hidden: number }) {
  return http({
    url: '/menu/updateHidden/' + id,
    method: 'post',
    params: params,
  })
}
