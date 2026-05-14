import http from '@/utils/http';
export function getOrderListAPI(params) {
    return http({
        url: '/order/list',
        method: 'get',
        params: params,
    });
}
export function orderUpdateCloseAPI(params) {
    return http({
        url: '/order/update/close',
        method: 'post',
        params: params,
    });
}
export function orderDeleteByIdsAPI(params) {
    return http({
        url: '/order/delete',
        method: 'post',
        params: params,
    });
}
export function orderUpdateDeliveryAPI(data) {
    return http({
        url: '/order/update/delivery',
        method: 'post',
        data: data,
    });
}
export function getOrderDetailByIdAPI(id) {
    return http({
        url: '/order/' + id,
        method: 'get',
    });
}
export function orderUpdateReceiverInfoAPI(data) {
    return http({
        url: '/order/update/receiverInfo',
        method: 'post',
        data: data,
    });
}
export function orderUpdateMoneyInfoAPI(data) {
    return http({
        url: '/order/update/moneyInfo',
        method: 'post',
        data: data,
    });
}
export function orderUpdateNoteAPI(params) {
    return http({
        url: '/order/update/note',
        method: 'post',
        params: params,
    });
}
