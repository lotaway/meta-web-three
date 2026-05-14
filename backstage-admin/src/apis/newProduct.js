import http from '@/utils/http';
export function getHomeNewProductListAPI(params) {
    return http({
        url: '/home/newProduct/list',
        method: 'get',
        params: params,
    });
}
export function homeNewProductUpdateRecommendStatusAPI(params) {
    return http({
        url: '/home/newProduct/update/recommendStatus',
        method: 'post',
        params: params,
    });
}
export function homeNewProductDeleteByIdsAPI(params) {
    return http({
        url: '/home/newProduct/delete',
        method: 'post',
        params: params,
    });
}
export function homeNewProductCreateAPI(data) {
    return http({
        url: '/home/newProduct/create',
        method: 'post',
        data: data,
    });
}
export function homeNewProductUpdateSortByIdAPI(params) {
    return http({
        url: '/home/newProduct/update/sort/' + params.id,
        method: 'post',
        params: params,
    });
}
