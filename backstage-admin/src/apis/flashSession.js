import http from '@/utils/http';
export function getFlashSessionSelectListAPI(params) {
    return http({
        url: '/flashSession/selectList',
        method: 'get',
        params: params,
    });
}
export function getFlashSessionListAPI() {
    return http({
        url: '/flashSession/list',
        method: 'get',
    });
}
export function flashSessionUpdateStatusByIdAPI(id, params) {
    return http({
        url: '/flashSession/update/status/' + id,
        method: 'post',
        params: params,
    });
}
export function flashSessionDeleteByIdAPI(id) {
    return http({
        url: '/flashSession/delete/' + id,
        method: 'post',
    });
}
export function flashSessionCreateAPI(data) {
    return http({
        url: '/flashSession/create',
        method: 'post',
        data: data,
    });
}
export function flashSessionUpdateByIdAPI(id, data) {
    return http({
        url: '/flashSession/update/' + id,
        method: 'post',
        data: data,
    });
}
