import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { OmsOrder } from '@/types/order'

export const useOrderStore = defineStore('order', () => {
  // Order list temporarily stored during batch shipment (for passing data between pages)
  const deliverOrderList = ref<OmsOrder[]>([])

  // Set the order list to pass
  const setDeliverOrderList = (list: OmsOrder[]) => {
    deliverOrderList.value = list
  }

  return {
    deliverOrderList,
    setDeliverOrderList,
  }
})
