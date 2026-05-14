import http from '@/utils/http';
export function getPrefrenceAreaListAllAPI() {
    return http({
        url: '/prefrenceArea/listAll',
        method: 'get',
    });
}
