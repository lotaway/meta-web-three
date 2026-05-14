import http from '@/utils/http';
export function getFlashProductRelationListAPI(params) {
    return http({
        url: '/flashProductRelation/list',
        method: 'get',
        params: params,
    });
}
export function flashProductRelationCreateAPI(data) {
    return http({
        url: '/flashProductRelation/create',
        method: 'post',
        data: data,
    });
}
export function flashProductRelationDeleteByIdAPI(id) {
    return http({
        url: '/flashProductRelation/delete/' + id,
        method: 'post',
    });
}
export function flashProductRelationUpdateByIdAPI(id, data) {
    return http({
        url: '/flashProductRelation/update/' + id,
        method: 'post',
        data: data,
    });
}
