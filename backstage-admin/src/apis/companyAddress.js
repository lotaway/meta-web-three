import http from '@/utils/http';
export function getCompanyAddressListAPI() {
    return http({
        url: '/companyAddress/list',
        method: 'get',
    });
}
