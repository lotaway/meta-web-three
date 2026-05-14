import http from '@/utils/http';
export function getReturnApplyListAPI(params) {
    return http({
        url: '/returnApply/list',
        method: 'get',
        params: params,
    });
}
export function returnApplyDeleteByIdsAPI(params) {
    return http({
        url: '/returnApply/delete',
        method: 'post',
        params: params,
    });
}
export function returnApplyUpdateStatusAPI(id, data) {
    return http({
        url: '/returnApply/update/status/' + id,
        method: 'post',
        data: data,
    });
}
export function getReturnApplyByIdAPI(id) {
    return http({
        url: '/returnApply/' + id,
        method: 'get',
    });
}
