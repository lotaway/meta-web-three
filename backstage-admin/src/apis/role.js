import http from '@/utils/http';
export function getRoleListAllAPI() {
    return http({
        url: '/role/listAll',
        method: 'get',
    });
}
export function getRoleListAPI(params) {
    return http({
        url: '/role/list',
        method: 'get',
        params: params,
    });
}
export function roleCreateAPI(data) {
    return http({
        url: '/role/create',
        method: 'post',
        data: data,
    });
}
export function roleUpdateByIdAPI(id, data) {
    return http({
        url: '/role/update/' + id,
        method: 'post',
        data: data,
    });
}
export function roleUpdateStatusAPI(id, params) {
    return http({
        url: '/role/updateStatus/' + id,
        method: 'post',
        params: params,
    });
}
export function roleDeleteByIdsAPI(params) {
    return http({
        url: '/role/delete',
        method: 'post',
        params: params,
    });
}
export function roleListMenuByRoleIdAPI(id) {
    return http({
        url: '/role/listMenu/' + id,
        method: 'get',
    });
}
export function roleAllocMenuAPI(params) {
    return http({
        url: '/role/allocMenu',
        method: 'post',
        params: params,
    });
}
export function roleListResourceById(id) {
    return http({
        url: '/role/listResource/' + id,
        method: 'get',
    });
}
export function roleAllocResourceAPI(params) {
    return http({
        url: '/role/allocResource',
        method: 'post',
        params: params,
    });
}
