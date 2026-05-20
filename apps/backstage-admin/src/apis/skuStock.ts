import type { PmsSkuStock } from '@/types/skuStock'
import http from '@/utils/http'
export function getSkuListByPidAPI(pid: number, params: { keyword?: string }) {
  return http<PmsSkuStock[]>({
    url: '/sku/' + pid,
    method: 'get',
    params: params,
  })
}
export function skuUpdateByPidAPI(pid: number, data: PmsSkuStock[]) {
  return http({
    url: '/sku/update/' + pid,
    method: 'post',
    data: data,
  })
}
