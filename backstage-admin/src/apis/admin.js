import http from '@/utils/http';
export function adminLoginAPI(data) {
    return http({
        method: 'POST',
        url: '/admin/login',
        data: data,
    });
}
export function adminLogoutAPI() {
    return http({
        method: 'POST',
        url: '/admin/logout',
    });
}
export function getAdminInfoAPI() {
    return http({
        method: 'GET',
        url: '/admin/info',
    });
}
export function getAdminListAPI(params) {
    return http({
        url: '/admin/list',
        method: 'get',
        params: params,
    });
}
export function adminRegisterAPI(data) {
    return http({
        url: '/admin/register',
        method: 'post',
        data: data,
    });
}
export function adminUpdateByIdAPI(id, data) {
    return http({
        url: '/admin/update/' + id,
        method: 'post',
        data: data,
    });
}
export function adminUpdateStatusByIdAPI(id, params) {
    return http({
        url: '/admin/updateStatus/' + id,
        method: 'post',
        params: params,
    });
}
export function adminDeleteByIdAPI(id) {
    return http({
        url: '/admin/delete/' + id,
        method: 'post',
    });
}
export function getRoleByAdminIdAPI(id) {
    return http({
        url: '/admin/role/' + id,
        method: 'get',
    });
}
export function adminRoleUpdateAPI(params) {
    return http({
        url: '/admin/role/update',
        method: 'post',
        params: params,
    });
}
