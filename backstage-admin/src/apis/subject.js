import http from '@/utils/http';
export function getSubjectListAllAPI() {
    return http({
        url: '/subject/listAll',
        method: 'get',
    });
}
export function getSubjectListAPI(params) {
    return http({
        url: '/subject/list',
        method: 'get',
        params: params,
    });
}
