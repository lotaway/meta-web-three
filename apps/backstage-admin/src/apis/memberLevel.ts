import type { UmsMemberLevel } from '@/types/memberLevel'
import http from '@/utils/http'

export function getMemberLevelListAPI(params?: { defaultStatus?: number }) {
  return http<UmsMemberLevel[]>({
    url: '/memberLevel/list',
    method: 'get',
    params: params,
  })
}

export function getMemberLevelByIdAPI(id: number) {
  return http<UmsMemberLevel>({
    url: `/memberLevel/${id}`,
    method: 'get',
  })
}

export function createMemberLevelAPI(data: UmsMemberLevel) {
  return http<number>({
    url: '/memberLevel',
    method: 'post',
    data,
  })
}

export function updateMemberLevelAPI(id: number, data: UmsMemberLevel) {
  return http<void>({
    url: `/memberLevel/${id}`,
    method: 'put',
    data,
  })
}

export function deleteMemberLevelAPI(id: number) {
  return http<void>({
    url: `/memberLevel/${id}`,
    method: 'delete',
  })
}