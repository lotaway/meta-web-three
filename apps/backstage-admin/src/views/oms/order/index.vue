<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Tickets } from '@element-plus/icons-vue'
import { getOrderListAPI, orderUpdateCloseAPI, orderDeleteByIdsAPI } from '@/apis/order'
import LogisticsDialog from '@/views/oms/order/components/logisticsDialog.vue'
import { formatDateTime } from '@/utils/datetime'
import type { OmsOrder, OrderQueryParam } from '@/types/order'
import { useOrderStore } from '@/stores/order'

const { t } = useI18n()

// Get route object
const router = useRouter()

// Get order store
const orderStore = useOrderStore()

// Order list query params
const listQuery = ref<OrderQueryParam>({
  pageNum: 1,
  pageSize: 10
})
// Order list data
const list = ref<OmsOrder[]>([])
// Table data loading progress bar
const listLoading = ref(true)
// Pagination component params
const total = ref(0)
// Get order list data
const getList = async () => {
  listLoading.value = true
  try {
    const response = await getOrderListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data.list
    total.value = response.data.total
  } catch (error) {
    listLoading.value = false
    ElMessage.error(t('oms.order.list.loadFailed'))
  }
}

// Initialize list info after mount
onMounted(() => {
  getList()
})

// Selected rows in the table
const multipleSelection = ref<OmsOrder[]>([])
// Batch operation type
const operateType = ref<number>()

// Close-order dialog data
const closeOrderData = ref({
  dialogVisible: false,
  content: '',
  orderIds: [] as number[]
})

// Logistics dialog visibility
const logisticsDialogVisible = ref(false)

// Order status options
const statusOptions = [
  {
    label: t('oms.order.list.statusPendingPay'),
    value: 0
  },
  {
    label: t('oms.order.list.statusPendingDeliver'),
    value: 1
  },
  {
    label: t('oms.order.list.statusDelivered'),
    value: 2
  },
  {
    label: t('oms.order.list.statusCompleted'),
    value: 3
  },
  {
    label: t('oms.order.list.statusClosed'),
    value: 4
  }
]

// Order type options
const orderTypeOptions = [
  {
    label: t('oms.order.list.typeNormal'),
    value: 0
  },
  {
    label: t('oms.order.list.typeFlash'),
    value: 1
  }
]

// Order source options
const sourceTypeOptions = [
  {
    label: t('oms.order.list.sourcePC'),
    value: 0
  },
  {
    label: t('oms.order.list.sourceAPP'),
    value: 1
  }
]

// Batch operation options
const operateOptions = [
  {
    label: t('oms.order.list.batchDeliver'),
    value: 1
  },
  {
    label: t('oms.order.list.batchClose'),
    value: 2
  },
  {
    label: t('oms.order.list.batchDelete'),
    value: 3
  }
]

// Format payment method
const formatPayType = (value: number) => {
  if (value === 1) {
    return t('oms.order.list.payAlipay')
  } else if (value === 2) {
    return t('oms.order.list.payWechat')
  } else {
    return t('oms.order.list.payUnpaid')
  }
}

// Format order source
const formatSourceType = (value: number) => {
  if (value === 1) {
    return t('oms.order.list.sourceAPP')
  } else {
    return t('oms.order.list.sourcePC')
  }
}

// Format order status
const formatStatus = (value: number) => {
  if (value === 1) {
    return t('oms.order.list.statusPendingDeliver')
  } else if (value === 2) {
    return t('oms.order.list.statusDelivered')
  } else if (value === 3) {
    return t('oms.order.list.statusCompleted')
  } else if (value === 4) {
    return t('oms.order.list.statusClosed')
  } else if (value === 5) {
    return t('oms.order.list.statusInvalid')
  } else {
    return t('oms.order.list.statusPendingPay')
  }
}

// Handle reset search
const handleResetSearch = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
}

// Handle search list
const handleSearchList = () => {
  listQuery.value.pageNum = 1
  getList()
}

// Handle table selection state change
const handleSelectionChange = (val: OmsOrder[]) => {
  multipleSelection.value = val
}

// Handle view order
const handleViewOrder = (index: number, row: OmsOrder) => {
  router.push({ path: '/oms/orderDetail', query: { id: row.id } })
}

// Handle close order
const handleCloseOrder = (index: number, row: OmsOrder) => {
  closeOrderData.value.dialogVisible = true
  closeOrderData.value.orderIds = [row.id!]
}

// Handle order shipment
const handleDeliveryOrder = (index: number, row: OmsOrder) => {
  orderStore.setDeliverOrderList([row])
  router.push({ path: '/oms/deliverOrderList' })
}

// Handle view logistics
const handleViewLogistics = (index: number, row: OmsOrder) => {
  logisticsDialogVisible.value = true
}

// Handle delete order
const handleDeleteOrder = async (index: number, row: OmsOrder) => {
  const ids = [row.id!]
  await deleteOrderFn(ids)
}

// Handle batch operations
const batchDeliver = () => {
  const listItems = multipleSelection.value.filter(item => item.status === 1)
  if (listItems.length < 1) {
    ElMessage({ message: t('oms.order.list.noDeliverable'), type: 'warning', duration: 1000 })
    return
  }
  orderStore.setDeliverOrderList(listItems)
  router.push({ path: '/oms/deliverOrderList' })
}

const batchClose = () => {
  closeOrderData.value.orderIds = multipleSelection.value.filter(item => item.status === 0).map(item => item.id)
  closeOrderData.value.dialogVisible = true
}

const batchDelete = async () => {
  const ids = multipleSelection.value.filter(item => item.status === 4).map(item => item.id)
  await deleteOrderFn(ids)
}

const batchHandlers: Record<number, () => void | Promise<void>> = {
  1: batchDeliver,
  2: batchClose,
  3: batchDelete,
}

const handleBatchOperate = async () => {
  if (!multipleSelection.value || multipleSelection.value.length < 1) {
    ElMessage({ message: t('oms.order.list.selectOrders'), type: 'warning', duration: 1000 })
    return
  }
  const handler = operateType.value != null ? batchHandlers[operateType.value] : undefined
  await handler?.()
}

// Handle page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Handle current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Handle confirm close order
const handleCloseOrderConfirm = async () => {
  if (!closeOrderData.value.content) {
    ElMessage({
      message: t('oms.order.list.noteRequired'),
      type: 'warning',
      duration: 1000
    })
    return
  }
  const orderIds = closeOrderData.value.orderIds.join(',')
  await orderUpdateCloseAPI({ ids: orderIds, note: closeOrderData.value.content })
  closeOrderData.value.orderIds = []
  closeOrderData.value.dialogVisible = false
  getList()
  ElMessage({
    message: t('oms.order.list.modifySuccess'),
    type: 'success',
    duration: 1000
  })
}

// Delete order function
const deleteOrderFn = async (ids: number[]) => {
  await ElMessageBox.confirm(t('oms.order.list.deleteConfirm'), t('common.warning'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await orderDeleteByIdsAPI({ ids: ids.join(',') })
  ElMessage({
    message: t('oms.order.list.deleteSuccess'),
    type: 'success',
    duration: 1000
  })
  getList()
}
</script>

<template>
  <div class="app-container">
    <el-card class="filter-container" shadow="never">
      <div>
        <el-icon class="el-icon-middle">
          <Search />
        </el-icon>
        <span>{{ t('oms.order.list.filterSearch') }}</span>
        <el-button style="float:right" type="primary" @click="handleSearchList()">
          {{ t('oms.order.list.querySearch') }}
        </el-button>
        <el-button style="float:right;margin-right: 15px" @click="handleResetSearch()">
          {{ t('oms.order.list.reset') }}
        </el-button>
      </div>
      <div style="margin-top: 20px">
        <el-form :inline="true" :model="listQuery" label-width="140px">
          <el-form-item :label="t('oms.order.list.inputSearch') + '：'">
            <el-input v-model="listQuery.orderSn" class="input-width" :placeholder="t('oms.order.list.orderSnPlaceholder')"></el-input>
          </el-form-item>
          <el-form-item :label="t('oms.order.list.receiver') + '：'">
            <el-input v-model="listQuery.receiverKeyword" class="input-width" :placeholder="t('oms.order.list.receiverPlaceholder')"></el-input>
          </el-form-item>
          <el-form-item :label="t('oms.order.list.createTime') + '：'">
            <el-date-picker class="input-width" v-model="listQuery.createTime" value-format="yyyy-MM-dd" type="date"
              :placeholder="t('oms.order.list.createTimePlaceholder')">
            </el-date-picker>
          </el-form-item>
          <el-form-item :label="t('oms.order.list.status') + '：'">
            <el-select v-model="listQuery.status" class="input-width" :placeholder="t('oms.order.list.all')" clearable>
              <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item :label="t('oms.order.list.orderType') + '：'">
            <el-select v-model="listQuery.orderType" class="input-width" :placeholder="t('oms.order.list.all')" clearable>
              <el-option v-for="item in orderTypeOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
          <el-form-item :label="t('oms.order.list.sourceType') + '：'">
            <el-select v-model="listQuery.sourceType" class="input-width" :placeholder="t('oms.order.list.all')" clearable>
              <el-option v-for="item in sourceTypeOptions" :key="item.value" :label="item.label" :value="item.value">
              </el-option>
            </el-select>
          </el-form-item>
        </el-form>
      </div>
    </el-card>
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('oms.order.list.dataList') }}</span>
    </el-card>
    <div class="table-container">
      <el-table ref="orderTable" :data="list" style="width: 100%;" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('oms.order.list.id')" width="80" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.orderNo')" width="180" align="center">
          <template #default="scope">{{ scope.row.orderSn }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.submitTime')" width="180" align="center">
          <template #default="scope">{{ formatDateTime(scope.row.createTime) }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.userAccount')" align="center">
          <template #default="scope">{{ scope.row.memberUsername }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.orderAmount')" width="120" align="center">
          <template #default="scope">￥{{ scope.row.totalAmount }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.payType')" width="120" align="center">
          <template #default="scope">{{ formatPayType(scope.row.payType) }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.orderSource')" width="120" align="center">
          <template #default="scope">{{ formatSourceType(scope.row.sourceType) }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.status')" width="120" align="center">
          <template #default="scope">{{ formatStatus(scope.row.status) }}</template>
        </el-table-column>
        <el-table-column :label="t('oms.order.list.operation')" width="200" align="center">
          <template #default="scope">
            <el-button size="small" @click="handleViewOrder(scope.$index, scope.row as OmsOrder)">{{ t('oms.order.list.viewOrder') }}</el-button>
            <el-button size="small" @click="handleCloseOrder(scope.$index, scope.row as OmsOrder)"
              v-show="scope.row.status === 0">{{ t('oms.order.list.closeOrder') }}</el-button>
            <el-button size="small" @click="handleDeliveryOrder(scope.$index, scope.row as OmsOrder)"
              v-show="scope.row.status === 1">{{ t('oms.order.list.deliverOrder') }}</el-button>
            <el-button size="small" @click="handleViewLogistics(scope.$index, scope.row as OmsOrder)"
              v-show="scope.row.status === 2 || scope.row.status === 3">{{ t('oms.order.list.trackOrder') }}</el-button>
            <el-button size="small" type="danger" @click="handleDeleteOrder(scope.$index, scope.row as OmsOrder)"
              v-show="scope.row.status === 4">{{ t('oms.order.list.deleteOrder') }}</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" :placeholder="t('oms.order.list.batchOperate')">
        <el-option v-for="item in operateOptions" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        {{ t('oms.order.list.confirm') }}
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" v-model:current-page="listQuery.pageNum"
        :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]" :total="total">
      </el-pagination>
    </div>
    <el-dialog :title="t('oms.order.list.closeOrderDialog')" v-model="closeOrderData.dialogVisible" width="30%">
      <span style="vertical-align: top">{{ t('oms.order.list.operationNote') }}：</span>
      <el-input style="width: 80%" type="textarea" :rows="5" :placeholder="t('oms.order.list.notePlaceholder')" v-model="closeOrderData.content">
      </el-input>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="closeOrderData.dialogVisible = false">{{ t('oms.order.list.cancel') }}</el-button>
          <el-button type="primary" @click="handleCloseOrderConfirm">{{ t('oms.order.list.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
    <logistics-dialog v-model="logisticsDialogVisible"></logistics-dialog>
  </div>
</template>

<style scoped>
.input-width {
  width: 203px
}
</style>
