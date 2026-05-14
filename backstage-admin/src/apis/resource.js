import http from '@/utils/http';
export function fetchAllResourceList() {
    return http({
        url: '/resource/listAll',
        method: 'get',
    });
}
export function getResourceListAPI(params) {
    return http({
        url: '/resource/list',
        method: 'get',
        params: params,
    });
}
export function resourceCreateAPI(data) {
    return http({
        url: '/resource/create',
        method: 'post',
        data: data,
    });
}
export function resourceUpdateAPI(id, data) {
    return http({
        url: '/resource/update/' + id,
        method: 'post',
        data: data,
    });
}
export function resourceDeleteByIdAPI(id) {
    return http({
        url: '/resource/delete/' + id,
        method: 'post',
    });
}
