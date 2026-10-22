import type { CommonPage, PageParam } from '@/types/common'
import type { PmsProductCategory, PmsProductCategoryExt } from '@/types/productCate'
import http from '@/utils/http'
export function getProductCategoryListWithChildrenAPI() {
  return http<PmsProductCategoryExt[]>({
    url: '/product-category/list/withChildren',
    method: 'get',
  })
}
export function getProductCategoryListAPI(parentId: number, params: PageParam) {
  return http<CommonPage<PmsProductCategory>>({
    url: '/product-category/list/' + parentId,
    method: 'get',
    params: params,
  })
}
export function productCategoryDeleteByIdAPI(id: number) {
  return http({
    url: '/product-category/delete/' + id,
    method: 'post',
  })
}
export function productCategoryCreateAPI(data: PmsProductCategory) {
  return http({
    url: '/product-category/create',
    method: 'post',
    data: data,
  })
}
export function productCategoryUpdateByIdAPI(id: number, data: PmsProductCategory) {
  return http({
    url: '/product-category/update/' + id,
    method: 'post',
    data: data,
  })
}
export function getProductCategoryByIdAPI(id: number) {
  return http<PmsProductCategory>({
    url: '/product-category/' + id,
    method: 'get',
  })
}
export function productCategoryUpdateShowStatusAPI(params: { ids: string; showStatus: number }) {
  return http({
    url: '/product-category/update/showStatus',
    method: 'post',
    params: params,
  })
}
export function productCategoryUpdateNavStatusAPI(params: { ids: string; navStatus: number }) {
  return http({
    url: '/product-category/update/navStatus',
    method: 'post',
    params: params,
  })
}
