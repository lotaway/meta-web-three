import http from '@/utils/http';
export function getHomeBrandListAPI(params) {
    return http({
        url: '/home/brand/list',
        method: 'get',
        params: params,
    });
}
export function homeBrandUpdateRecommendStatusAPI(params) {
    return http({
        url: '/home/brand/update/recommendStatus',
        method: 'post',
        params: params,
    });
}
export function homeBrandDeleteByIdsAPI(params) {
    return http({
        url: '/home/brand/delete',
        method: 'post',
        params: params,
    });
}
export function homeBrandCreateAPI(data) {
    return http({
        url: '/home/brand/create',
        method: 'post',
        data: data,
    });
}
export function homeBrandUpdateSortAPI(params) {
    return http({
        url: '/home/brand/update/sort/' + params.id,
        method: 'post',
        params: params,
    });
}
