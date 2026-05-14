import http from '@/utils/http';
export function resourceCategoryListAllAPI() {
    return http({
        url: '/resourceCategory/listAll',
        method: 'get',
    });
}
export function resourceCategoryCreateAPI(data) {
    return http({
        url: '/resourceCategory/create',
        method: 'post',
        data: data,
    });
}
export function resourceCategoryUpdateAPI(id, data) {
    return http({
        url: '/resourceCategory/update/' + id,
        method: 'post',
        data: data,
    });
}
export function resourceCategoryDeleteByIdAPI(id) {
    return http({
        url: '/resourceCategory/delete/' + id,
        method: 'post',
    });
}
