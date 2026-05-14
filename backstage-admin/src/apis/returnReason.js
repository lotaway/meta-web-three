import http from '@/utils/http';
export function getReturnReasonListAPI(params) {
    return http({
        url: '/returnReason/list',
        method: 'get',
        params: params,
    });
}
export function returnReasonDeleteByIdsAPI(params) {
    return http({
        url: '/returnReason/delete',
        method: 'post',
        params: params,
    });
}
export function returnReasonUpdateStatusAPI(params) {
    return http({
        url: '/returnReason/update/status',
        method: 'post',
        params: params,
    });
}
export function returnReasonCreateAPI(data) {
    return http({
        url: '/returnReason/create',
        method: 'post',
        data: data,
    });
}
export function getReturnReasonByIdAPI(id) {
    return http({
        url: '/returnReason/' + id,
        method: 'get',
    });
}
export function returnReasonUpdateAPI(id, data) {
    return http({
        url: '/returnReason/update/' + id,
        method: 'post',
        data: data,
    });
}
