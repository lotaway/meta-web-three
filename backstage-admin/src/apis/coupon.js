import http from '@/utils/http';
export function getCouponListAPI(params) {
    return http({
        url: '/coupon/list',
        method: 'get',
        params: params,
    });
}
export function couponCreateAPI(data) {
    return http({
        url: '/coupon/create',
        method: 'post',
        data: data,
    });
}
export function getCouponByIdAPI(id) {
    return http({
        url: '/coupon/' + id,
        method: 'get',
    });
}
export function couponUpdateByIdAPI(id, data) {
    return http({
        url: '/coupon/update/' + id,
        method: 'post',
        data: data,
    });
}
export function couponDeleteByIdAPI(id) {
    return http({
        url: '/coupon/delete/' + id,
        method: 'post',
    });
}
export function getCouponHistoryListAPI(params) {
    return http({
        url: '/couponHistory/list',
        method: 'get',
        params: params,
    });
}
