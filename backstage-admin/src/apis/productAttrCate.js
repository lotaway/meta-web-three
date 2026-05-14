import http from '@/utils/http';
export function productAttributeCategoryListWithAttrAPI() {
    return http({
        url: '/productAttribute/category/list/withAttr',
        method: 'get',
    });
}
export function getProductAttributeCategoryListAPI(params) {
    return http({
        url: '/productAttribute/category/list',
        method: 'get',
        params: params,
    });
}
export function productAttributeCategoryCreateAPI(name) {
    return http({
        url: '/productAttribute/category/create',
        method: 'post',
        params: { name: name },
    });
}
export function productAttributeCategoryDeleteById(id) {
    return http({
        url: '/productAttribute/category/delete/' + id,
        method: 'get',
    });
}
export function productAttributeCategoryUpdateAPI(id, name) {
    return http({
        url: '/productAttribute/category/update/' + id,
        method: 'post',
        params: { name: name },
    });
}
