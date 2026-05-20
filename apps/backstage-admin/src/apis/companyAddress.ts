import type { OmsCompanyAddress } from '@/types/companyAddress'
import http from '@/utils/http'
export function getCompanyAddressListAPI() {
  return http<OmsCompanyAddress[]>({
    url: '/companyAddress/list',
    method: 'get',
  })
}
