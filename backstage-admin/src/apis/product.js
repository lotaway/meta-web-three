import http from '@/utils/http';
export function getProductListAPI(params) {
    return http({
        url: '/product/list',
        method: 'get',
        params: params,
    });
}
export function productUpdateDeleteStatusAPI(params) {
    return http({
        url: '/product/update/deleteStatus',
        method: 'post',
        params: params,
    });
}
export function productUpdateNewStatusAPI(params) {
    return http({
        url: '/product/update/newStatus',
        method: 'post',
        params: params,
    });
}
export function productUpdateRecommendStatusAPI(params) {
    return http({
        url: '/product/update/recommendStatus',
        method: 'post',
        params: params,
    });
}
export function productUpdatePublishStatusAPI(params) {
    return http({
        url: '/product/update/publishStatus',
        method: 'post',
        params: params,
    });
}
export function productCreateAPI(data) {
    return http({
        url: '/product/create',
        method: 'post',
        data: data,
    });
}
export function productUpdateByIdAPI(id, data) {
    return http({
        url: '/product/update/' + id,
        method: 'post',
        data: data,
    });
}
export function getPruductUpdateInfoAPI(id) {
    return http({
        url: '/product/updateInfo/' + id,
        method: 'get',
    });
}
