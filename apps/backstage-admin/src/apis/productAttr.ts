import type { CommonPage, PageParam } from '@/types/common'
import type { PmsProductAttribute, ProductAttrInfo } from '@/types/productAttr'
import http from '@/utils/http'
export function getProductAttrInfoByCateIdAPI(cateId: number) {
  return http<ProductAttrInfo[]>({
    url: '/productAttribute/attrInfo/' + cateId,
    method: 'get',
  })
}
export function getProductAttributeListAPI(cid: number, params: PageParam & { type: number }) {
  return http<CommonPage<PmsProductAttribute>>({
    url: '/productAttribute/list/' + cid,
    method: 'get',
    params: params,
  })
}
export function productAttributeDeleteByIds(params: { ids: string }) {
  return http({
    url: '/productAttribute/delete',
    method: 'post',
    params: params,
  })
}
export function productAttributeCreateAPI(data: PmsProductAttribute) {
  return http({
    url: '/productAttribute/create',
    method: 'post',
    data: data,
  })
}
export function productAttributeUpdateAPI(id: number, data: PmsProductAttribute) {
  return http({
    url: '/productAttribute/update/' + id,
    method: 'post',
    data: data,
  })
}
export function getProductAttributeByIdAPI(id: number) {
  return http<PmsProductAttribute>({
    url: '/productAttribute/' + id,
    method: 'get',
  })
}
