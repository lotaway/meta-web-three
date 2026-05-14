import http from '@/utils/http';
export function getHomeAdvertiseListAPI(params) {
    return http({
        url: '/home/advertise/list',
        method: 'get',
        params: params,
    });
}
export function homeAdvertiseUpdateStatusAPI(params) {
    return http({
        url: '/home/advertise/update/status/' + params.id,
        method: 'post',
        params: params,
    });
}
export function deleteHomeAdvertiseAPI(params) {
    return http({
        url: '/home/advertise/delete',
        method: 'post',
        params: params,
    });
}
export function homeAdvertiseCreateAPI(data) {
    return http({
        url: '/home/advertise/create',
        method: 'post',
        data: data,
    });
}
export function getHomeAdvertiseByIdAPI(id) {
    return http({
        url: '/home/advertise/' + id,
        method: 'get',
    });
}
export function homeAdvertiseUpdateAPI(id, data) {
    return http({
        url: '/home/advertise/update/' + id,
        method: 'post',
        data: data,
    });
}
