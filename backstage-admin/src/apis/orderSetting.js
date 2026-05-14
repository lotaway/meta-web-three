import http from '@/utils/http';
export function getOrderSettingByIdAPI(id) {
    return http({
        url: '/orderSetting/' + id,
        method: 'get',
    });
}
export function orderSettingUpdateByIdAPI(id, data) {
    return http({
        url: '/orderSetting/update/' + id,
        method: 'post',
        data: data,
    });
}
