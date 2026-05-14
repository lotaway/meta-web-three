import http from '@/utils/http';
export function getHomeRecommendSubjectListAPI(params) {
    return http({
        url: '/home/recommendSubject/list',
        method: 'get',
        params: params,
    });
}
export function homeRecommendSubjectUpdateRecommendStatusAPI(params) {
    return http({
        url: '/home/recommendSubject/update/recommendStatus',
        method: 'post',
        params: params,
    });
}
export function homeRecommendSubjectDeleteByIdsAPI(params) {
    return http({
        url: '/home/recommendSubject/delete',
        method: 'post',
        params: params,
    });
}
export function homeRecommendSubjectCreateAPI(data) {
    return http({
        url: '/home/recommendSubject/create',
        method: 'post',
        data: data,
    });
}
export function homeRecommendSubjectUpdateSortAPI(params) {
    return http({
        url: '/home/recommendSubject/update/sort/' + params.id,
        method: 'post',
        params: params,
    });
}
