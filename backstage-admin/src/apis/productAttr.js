import http from '@/utils/http';
export function getProductAttrInfoByCateIdAPI(cateId) {
    return http({
        url: '/productAttribute/attrInfo/' + cateId,
        method: 'get',
    });
}
export function getProductAttributeListAPI(cid, params) {
    return http({
        url: '/productAttribute/list/' + cid,
        method: 'get',
        params: params,
    });
}
export function productAttributeDeleteByIds(params) {
    return http({
        url: '/productAttribute/delete',
        method: 'post',
        params: params,
    });
}
export function productAttributeCreateAPI(data) {
    return http({
        url: '/productAttribute/create',
        method: 'post',
        data: data,
    });
}
export function productAttributeUpdateAPI(id, data) {
    return http({
        url: '/productAttribute/update/' + id,
        method: 'post',
        data: data,
    });
}
export function getProductAttributeByIdAPI(id) {
    return http({
        url: '/productAttribute/' + id,
        method: 'get',
    });
}
