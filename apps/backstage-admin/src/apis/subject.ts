import type { CommonPage, PageParam } from '@/types/common'
import type { CmsSubject } from '@/types/subject'
import http from '@/utils/http'
export function getSubjectListAllAPI() {
  return http<CmsSubject[]>({
    url: '/subject/listAll',
    method: 'get',
  })
}
export function getSubjectListAPI(params: PageParam) {
  return http<CommonPage<CmsSubject>>({
    url: '/subject/list',
    method: 'get',
    params: params,
  })
}
