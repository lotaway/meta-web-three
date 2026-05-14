import http from '@/utils/http';
export function getFlashListAPI(params) {
    return http({
        url: '/flash/list',
        method: 'get',
        params: params,
    });
}
export function flashUpdateStatusByIdAPI(id, params) {
    return http({
        url: '/flash/update/status/' + id,
        method: 'post',
        params: params,
    });
}
export function flashDeleteByIdAPI(id) {
    return http({
        url: '/flash/delete/' + id,
        method: 'post',
    });
}
export function flashCreateAPI(data) {
    return http({
        url: '/flash/create',
        method: 'post',
        data: data,
    });
}
export function flashUpdateByIdAPI(id, data) {
    return http({
        url: '/flash/update/' + id,
        method: 'post',
        data: data,
    });
}
