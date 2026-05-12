import type { CommonPage, PageParam } from '@/types/common'
import type {
  PmsProductAttributeCategory,
  PmsProductAttributeCategoryExt,
} from '@/types/productAttr'
import http from '@/utils/http'
export function productAttributeCategoryListWithAttrAPI() {
  return http<PmsProductAttributeCategoryExt[]>({
    url: '/productAttribute/category/list/withAttr',
    method: 'get',
  })
}
export function getProductAttributeCategoryListAPI(params: PageParam) {
  return http<CommonPage<PmsProductAttributeCategory>>({
    url: '/productAttribute/category/list',
    method: 'get',
    params: params,
  })
}
export function productAttributeCategoryCreateAPI(name: string) {
  return http({
    url: '/productAttribute/category/create',
    method: 'post',
    params: { name: name },
  })
}
export function productAttributeCategoryDeleteById(id: number) {
  return http({
    url: '/productAttribute/category/delete/' + id,
    method: 'get',
  })
}
export function productAttributeCategoryUpdateAPI(id: number, name: string) {
  return http({
    url: '/productAttribute/category/update/' + id,
    method: 'post',
    params: { name: name },
  })
}
