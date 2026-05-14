import http from '@/utils/http';
export function getProductCategoryListWithChildrenAPI() {
    return http({
        url: '/productCategory/list/withChildren',
        method: 'get',
    });
}
export function getProductCategoryListAPI(parentId, params) {
    return http({
        url: '/productCategory/list/' + parentId,
        method: 'get',
        params: params,
    });
}
export function productCategoryDeleteByIdAPI(id) {
    return http({
        url: '/productCategory/delete/' + id,
        method: 'post',
    });
}
export function productCategoryCreateAPI(data) {
    return http({
        url: '/productCategory/create',
        method: 'post',
        data: data,
    });
}
export function productCategoryUpdateByIdAPI(id, data) {
    return http({
        url: '/productCategory/update/' + id,
        method: 'post',
        data: data,
    });
}
export function getProductCategoryByIdAPI(id) {
    return http({
        url: '/productCategory/' + id,
        method: 'get',
    });
}
export function productCategoryUpdateShowStatusAPI(params) {
    return http({
        url: '/productCategory/update/showStatus',
        method: 'post',
        params: params,
    });
}
export function productCategoryUpdateNavStatusAPI(params) {
    return http({
        url: '/productCategory/update/navStatus',
        method: 'post',
        params: params,
    });
}
