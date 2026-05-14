import http from '@/utils/http';
export function getHomeRecommendProductListAPI(params) {
    return http({
        url: '/home/recommendProduct/list',
        method: 'get',
        params: params,
    });
}
export function homeRecommendProductUpdateRecommendStatusAPI(params) {
    return http({
        url: '/home/recommendProduct/update/recommendStatus',
        method: 'post',
        params: params,
    });
}
export function homeRecommendProductDeleteByIdsAPI(params) {
    return http({
        url: '/home/recommendProduct/delete',
        method: 'post',
        params: params,
    });
}
export function homeRecommendProductCreateAPI(data) {
    return http({
        url: '/home/recommendProduct/create',
        method: 'post',
        data: data,
    });
}
export function homeRecommendProductUpdateSortByIdAPI(params) {
    return http({
        url: '/home/recommendProduct/update/sort/' + params.id,
        method: 'post',
        params: params,
    });
}
