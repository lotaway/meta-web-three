import type { UmsMemberLevel } from '@/types/memberLevel'
import http from '@/utils/http'
export function getMemberLevelListAPI(params: { defaultStatus: number }) {
  return http<UmsMemberLevel[]>({
    url: '/memberLevel/list',
    method: 'get',
    params: params,
  })
}
