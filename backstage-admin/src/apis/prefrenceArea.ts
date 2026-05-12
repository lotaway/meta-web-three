import type { CmsPrefrenceArea } from '@/types/prefrenceArea'
import http from '@/utils/http'
export function getPrefrenceAreaListAllAPI() {
  return http<CmsPrefrenceArea[]>({
    url: '/prefrenceArea/listAll',
    method: 'get',
  })
}
