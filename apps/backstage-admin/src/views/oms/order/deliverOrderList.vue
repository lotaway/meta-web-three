<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Tickets } from '@element-plus/icons-vue'
import { orderUpdateDeliveryAPI } from '@/apis/order'
import type { OmsOrder, OmsOrderDeliveryParam } from '@/types/order'
import { useOrderStore } from '@/stores/order'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const router = useRouter()
// Get order store
const orderStore = useOrderStore()

// Default shipping company option
const defaultLogisticsCompanies = ["顺丰快递", "圆通快递", "中通快递", "韵达快递"]

// Shipment order list data
const list = ref<OmsOrder[]>([])
// Shipping company options
const companyOptions = ref<string[]>(defaultLogisticsCompanies)

// Get detailed address from the order object
const fortmatAddress = (order: OmsOrder) => {
  return order.receiverProvince + order.receiverCity + order.receiverRegion + order.receiverDetailAddress
}

// Initialize data after mount
onMounted(() => {
  list.value = orderStore.deliverOrderList
  // Clear data in the store
  orderStore.setDeliverOrderList([])
})

// Cancel operation
const cancel = () => {
  router.back()
}

// Confirm shipment operation
const buildDeliveryParams = (): OmsOrderDeliveryParam[] =>
  list.value.map(item => ({
    orderId: item.id,
    deliverySn: item.orderSn,
    deliveryCompany: item.deliveryCompany
  }))

const confirmDeliver = async () => {
  await ElMessageBox.confirm(t('dialog.confirmDeliver'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await orderUpdateDeliveryAPI(buildDeliveryParams())
  router.back()
  ElMessage({ type: 'success', message: '发货成功!' })
}

const confirm = async () => {
  try {
    await confirmDeliver()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage({ type: 'info', message: '已取消发货' })
    }
  }
}
</script>

<template>
  <div class="app-container">
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('oms.deliveryList') }}</span>
    </el-card>
    <div class="table-container">
      <el-table ref="deliverOrderTable" style="width: 100%;" :data="list" border>
        <el-table-column :label="t('oms.order.list.orderSn')" width="180" align="center">
          <template #default="scope">{{ scope.row.orderSn }}</template>
        </el-table-column>
        <el-table-column label="收货人" width="150" align="center">
          <template #default="scope">{{ scope.row.receiverName }}</template>
        </el-table-column>
        <el-table-column label="手机号码" width="160" align="center">
          <template #default="scope">{{ scope.row.receiverPhone }}</template>
        </el-table-column>
        <el-table-column label="邮政编码" width="160" align="center">
          <template #default="scope">{{ scope.row.receiverPostCode }}</template>
        </el-table-column>
        <el-table-column label="收货地址" align="center">
          <template #default="scope">{{ fortmatAddress(scope.row as OmsOrder) }}</template>
        </el-table-column>
        <el-table-column label="配送方式" width="200" align="center">
          <template #default="scope">
            <el-select :placeholder="t('oms.selectCarrier')" v-model="scope.row.deliveryCompany">
              <el-option v-for="item in companyOptions" :key="item" :label="item" :value="item">
              </el-option>
            </el-select>
          </template>
        </el-table-column>
        <el-table-column label="物流单号" width="180" align="center">
          <template #default="scope">
            <el-input v-model="scope.row.deliverySn"></el-input>
          </template>
        </el-table-column>
      </el-table>
      <div style="margin-top: 15px;text-align: center">
        <el-button @click="cancel">取消</el-button>
        <el-button @click="confirm" type="primary">确定</el-button>
      </div>
    </div>
  </div>
</template>

<style></style>
