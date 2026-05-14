import http from '@/utils/http';
export function getMenuTreeListAPI() {
    return http({
        url: '/menu/treeList',
        method: 'get',
    });
}
export function getMenuListByParentIdAPI(parentId, params) {
    return http({
        url: '/menu/list/' + parentId,
        method: 'get',
        params: params,
    });
}
export function deleteMenuByIdAPI(id) {
    return http({
        url: '/menu/delete/' + id,
        method: 'post',
    });
}
export function menuCreateAPI(data) {
    return http({
        url: '/menu/create',
        method: 'post',
        data: data,
    });
}
export function updateMenu(id, data) {
    return http({
        url: '/menu/update/' + id,
        method: 'post',
        data: data,
    });
}
export function getMenuByIdAPI(id) {
    return http({
        url: '/menu/' + id,
        method: 'get',
    });
}
export function menuUpdateHiddenByIdAPI(id, params) {
    return http({
        url: '/menu/updateHidden/' + id,
        method: 'post',
        params: params,
    });
}
