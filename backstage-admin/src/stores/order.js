import { defineStore } from 'pinia';
import { ref } from 'vue';
export const useOrderStore = defineStore('order', () => {
    // 批量发货时临时存储的订单列表（用于页面间传递数据）
    const deliverOrderList = ref([]);
    // 设置需要传递的订单列表
    const setDeliverOrderList = (list) => {
        deliverOrderList.value = list;
    };
    return {
        deliverOrderList,
        setDeliverOrderList,
    };
});
