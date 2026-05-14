import http from '@/utils/http';
export function getMemberLevelListAPI(params) {
    return http({
        url: '/memberLevel/list',
        method: 'get',
        params: params,
    });
}
