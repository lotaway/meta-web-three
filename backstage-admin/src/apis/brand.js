import http from '@/utils/http';
export function getBrandListAPI(params) {
    return http({
        url: '/brand/list',
        method: 'get',
        params: params,
    });
}
export function createBrandAPI(data) {
    return http({
        url: '/brand/create',
        method: 'post',
        data: data,
    });
}
export function brandUpdateShowStatusAPI(params) {
    return http({
        url: '/brand/update/showStatus',
        method: 'post',
        params: params,
    });
}
export function brandUpdateFactoryStatusAPI(params) {
    return http({
        url: '/brand/update/factoryStatus',
        method: 'post',
        params: params,
    });
}
export function brandDeleteByIdAPI(id) {
    return http({
        url: '/brand/delete/' + id,
        method: 'get',
    });
}
export function getBrandAPI(id) {
    return http({
        url: '/brand/' + id,
        method: 'get',
    });
}
export function updateBrandAPI(id, data) {
    return http({
        url: '/brand/update/' + id,
        method: 'post',
        data: data,
    });
}
