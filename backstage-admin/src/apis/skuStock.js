import http from '@/utils/http';
export function getSkuListByPidAPI(pid, params) {
    return http({
        url: '/sku/' + pid,
        method: 'get',
        params: params,
    });
}
export function skuUpdateByPidAPI(pid, data) {
    return http({
        url: '/sku/update/' + pid,
        method: 'post',
        data: data,
    });
}
