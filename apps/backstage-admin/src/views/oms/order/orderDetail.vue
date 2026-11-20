<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox, type CascaderOption } from 'element-plus'
import { Warning } from '@element-plus/icons-vue'
import { getOrderDetailByIdAPI, orderUpdateReceiverInfoAPI, orderUpdateMoneyInfoAPI, orderUpdateCloseAPI, orderUpdateNoteAPI, orderDeleteByIdsAPI } from '@/apis/order'
import LogisticsDialog from '@/views/oms/order/components/logisticsDialog.vue'
import type { OmsOrder, OmsOrderDetail, OmsReceiverInfoParam } from '@/types/order'
import { formatDateTime } from '@/utils/datetime'
import { pcaTextArr } from 'element-china-area-data'

const { t } = useI18n()

// Get route object
const router = useRouter()
const route = useRoute()

// Order ID
const id = ref<number>()
// Order detail data
const order = ref<OmsOrderDetail>({} as OmsOrderDetail)

// Get order details after mount
onMounted(async () => {
  id.value = Number(route.query.id)
  const res = await getOrderDetailByIdAPI(id.value!)
  order.value = res.data
})

// Edit-recipient dialog visibility
const receiverDialogVisible = ref(false)
// Recipient information
const receiverInfo = ref<OmsReceiverInfoParam>({} as OmsReceiverInfoParam)
// Currently selected province/city/district cascade
const selectedRegions = ref<string[]>([])

// Edit-fee dialog visibility
const moneyDialogVisible = ref(false)
// Fee information
const moneyInfo = ref({ orderId: 0, freightAmount: 0, discountAmount: 0, status: 0 })

// Send-internal-message dialog visibility
const messageDialogVisible = ref(false)
// Internal message content
const message = ref({ title: '', content: '' })

// Close-order dialog visibility
const closeDialogVisible = ref(false)
// Close-order information
const closeInfo = ref({ note: '', id: 0 })

// Order-remark dialog visibility
const markOrderDialogVisible = ref(false)
// Order-remark information
const markInfo = ref({ id: 0, note: '' })

// Logistics dialog visibility
const logisticsDialogVisible = ref(false)

// Format empty values
const formatNull = (value: string | undefined) => {
  if (!value) {
    return t('oms.order.detail.cellNoData')
  } else {
    return value
  }
}

// Format long text
const formatLongText = (value: string | undefined) => {
  if (!value) {
    return t('oms.order.detail.cellNoData')
  } else if (value.length > 8) {
    return value.substring(0, 8) + '...'
  } else {
    return value
  }
}

// Format payment method
const formatPayType = (value: number) => {
  if (value === 1) {
    return t('oms.order.detail.payTypeAlipay')
  } else if (value === 2) {
    return t('oms.order.detail.payTypeWechat')
  } else {
    return t('oms.order.detail.payTypeUnpaid')
  }
}

// Format order source
const formatSourceType = (value: number) => {
  if (value === 1) {
    return t('oms.order.detail.sourceTypeAPP')
  } else {
    return t('oms.order.detail.sourceTypePC')
  }
}

// Format order type
const formatOrderType = (value: number) => {
  if (value === 1) {
    return t('oms.order.detail.orderTypeFlash')
  } else {
    return t('oms.order.detail.orderTypeNormal')
  }
}

// Format address
const formatAddress = (order: OmsOrder) => {
  let str = order.receiverProvince || ''
  if (order.receiverCity != null) {
    str += "  " + order.receiverCity
  }
  str += "  " + (order.receiverRegion || '')
  str += "  " + (order.receiverDetailAddress || '')
  return str
}

// Format order status
const formatStatus = (value: number) => {
  if (value === 1) {
    return t('oms.order.detail.statusPendingDeliver')
  } else if (value === 2) {
    return t('oms.order.detail.statusDelivered')
  } else if (value === 3) {
    return t('oms.order.detail.statusCompleted')
  } else if (value === 4) {
    return t('oms.order.detail.statusClosed')
  } else if (value === 5) {
    return t('oms.order.detail.statusInvalid')
  } else {
    return t('oms.order.detail.statusPendingPay')
  }
}

// Format payment status
const formatPayStatus = (value: number) => {
  if (value === 0) {
    return t('oms.order.detail.payStatusUnpaid')
  } else if (value === 4) {
    return t('oms.order.detail.payStatusRefunded')
  } else {
    return t('oms.order.detail.payStatusPaid')
  }
}

// Format shipment status
const formatDeliverStatus = (value: number) => {
  if (value === 0 || value === 1) {
    return t('oms.order.detail.deliverStatusNotDelivered')
  } else {
    return t('oms.order.detail.deliverStatusDelivered')
  }
}

// Format product attributes
const formatProductAttr = (value: string) => {
  if (value == null) {
    return ''
  } else {
    const attr = JSON.parse(value)
    let result = ''
    for (let i = 0; i < attr.length; i++) {
      result += attr[i].key
      result += ":"
      result += attr[i].value
      result += ";"
    }
    return result
  }
}

// Format step status
const formatStepStatus = (value: number) => {
  if (value === 1) {
    // Pending shipment
    return 2
  } else if (value === 2) {
    // Shipped
    return 3
  } else if (value === 3) {
    // Completed
    return 4
  } else {
    // Pending payment, closed, invalid orders
    return 1
  }
}

// Select region
const onSelectRegionChange = () => {
  receiverInfo.value.receiverProvince = selectedRegions.value[0]
  receiverInfo.value.receiverCity = selectedRegions.value[1]
  receiverInfo.value.receiverRegion = selectedRegions.value[2]
}

// Show the edit-recipient dialog
const showUpdateReceiverDialog = () => {
  receiverDialogVisible.value = true
  receiverInfo.value = {
    orderId: order.value.id,
    receiverName: order.value.receiverName,
    receiverPhone: order.value.receiverPhone,
    receiverPostCode: order.value.receiverPostCode,
    receiverDetailAddress: order.value.receiverDetailAddress,
    receiverProvince: order.value.receiverProvince,
    receiverCity: order.value.receiverCity,
    receiverRegion: order.value.receiverRegion,
    status: order.value.status
  }
  // Initialize data in the address picker
  selectedRegions.value = []
  selectedRegions.value[0] = receiverInfo.value.receiverProvince!
  selectedRegions.value[1] = receiverInfo.value.receiverCity!
  selectedRegions.value[2] = receiverInfo.value.receiverRegion!
}

// Handle update recipient information
const handleUpdateReceiverInfo = async () => {
  await ElMessageBox.confirm(t('oms.order.detail.msgModifyReceiverConfirm'), t('oms.order.detail.confirmTitle'), {
    confirmButtonText: t('common.confirmText'),
    cancelButtonText: t('common.cancelText'),
    type: 'warning'
  })
  await orderUpdateReceiverInfoAPI(receiverInfo.value)
  receiverDialogVisible.value = false
  ElMessage({
    type: 'success',
    message: t('oms.order.detail.msgModifySuccess')
  })
  const response = await getOrderDetailByIdAPI(id.value!)
  order.value = response.data
}

// Show the edit-fee dialog
const showUpdateMoneyDialog = () => {
  moneyDialogVisible.value = true
  moneyInfo.value.orderId = order.value.id
  moneyInfo.value.freightAmount = order.value.freightAmount
  moneyInfo.value.discountAmount = order.value.discountAmount
  moneyInfo.value.status = order.value.status
}

// Handle update fee information
const handleUpdateMoneyInfo = async () => {
  await ElMessageBox.confirm(t('oms.order.detail.msgModifyMoneyConfirm'), t('oms.order.detail.confirmTitle'), {
    confirmButtonText: t('common.confirmText'),
    cancelButtonText: t('common.cancelText'),
    type: 'warning'
  })
  await orderUpdateMoneyInfoAPI(moneyInfo.value)
  moneyDialogVisible.value = false
  ElMessage({
    type: 'success',
    message: t('oms.order.detail.msgModifySuccess')
  })
  const response = await getOrderDetailByIdAPI(id.value!)
  order.value = response.data
}

// Show the internal-message dialog
const showMessageDialog = () => {
  messageDialogVisible.value = true
  message.value.title = ''
  message.value.content = ''
}

// Handle send internal message
const handleSendMessage = async () => {
  await ElMessageBox.confirm(t('oms.order.detail.msgSendMessageConfirm'), t('oms.order.detail.confirmTitle'), {
    confirmButtonText: t('common.confirmText'),
    cancelButtonText: t('common.cancelText'),
    type: 'warning'
  })
  console.log(t('oms.order.detail.msgFeatureNotImpl'))
  messageDialogVisible.value = false
  ElMessage({
    type: 'success',
    message: t('oms.order.detail.msgSendSuccess')
  })
}

// Show the close-order dialog
const showCloseOrderDialog = () => {
  closeDialogVisible.value = true
  closeInfo.value.note = ''
  closeInfo.value.id = id.value!
}

// Handle close order
const handleCloseOrder = async () => {
  try {
    await ElMessageBox.confirm(t('oms.order.detail.msgCloseOrderConfirm'), t('oms.order.detail.confirmTitle'), {
      confirmButtonText: t('common.confirmText'),
      cancelButtonText: t('common.cancelText'),
      type: 'warning'
    })
    await orderUpdateCloseAPI({ ids: closeInfo.value.id.toString(), note: closeInfo.value.note })
    closeDialogVisible.value = false
    ElMessage({
      type: 'success',
      message: t('oms.order.detail.msgCloseSuccess')
    })

    const response = await getOrderDetailByIdAPI(id.value!)
    order.value = response.data
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to close order:', error)
    }
  }
}

// Show the order-remark dialog
const showMarkOrderDialog = () => {
  markOrderDialogVisible.value = true
  markInfo.value.id = id.value!
  markInfo.value.note = ''
}

// Handle order remark
const handleMarkOrder = async () => {
  await ElMessageBox.confirm(t('oms.order.detail.msgMarkOrderConfirm'), t('oms.order.detail.confirmTitle'), {
    confirmButtonText: t('common.confirmText'),
    cancelButtonText: t('common.cancelText'),
    type: 'warning'
  })
  await orderUpdateNoteAPI({ id: markInfo.value.id, note: markInfo.value.note, status: order.value.status })
  markOrderDialogVisible.value = false
  ElMessage({
    type: 'success',
    message: t('oms.order.detail.msgMarkSuccess')
  })
  const response = await getOrderDetailByIdAPI(id.value!)
  order.value = response.data
}

// Handle delete order
const handleDeleteOrder = async () => {
  try {
    await ElMessageBox.confirm(t('oms.order.detail.msgDeleteOrderConfirm'), t('oms.order.detail.confirmTitle'), {
      confirmButtonText: t('common.confirmText'),
      cancelButtonText: t('common.cancelText'),
      type: 'warning'
    })
    await orderDeleteByIdsAPI({ ids: id.value!.toString() })
    ElMessage({
      message: t('oms.order.detail.msgDeleteSuccess'),
      type: 'success',
      duration: 1000
    })
    router.back()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete order:', error)
    }
  }
}

// Show the logistics dialog
const showLogisticsDialog = () => {
  logisticsDialogVisible.value = true
}

</script>

<template>
  <div class="detail-container">
    <div>
      <el-steps :active="formatStepStatus(order.status)" finish-status="success" align-center>
        <el-step :title="t('oms.order.detail.stepSubmitOrder')" :description="order.createTime ? formatDateTime(order.createTime) : ''"></el-step>
        <el-step :title="t('oms.order.detail.stepPayOrder')" :description="order.paymentTime ? formatDateTime(order.paymentTime) : ''"></el-step>
        <el-step :title="t('oms.order.detail.stepPlatformDeliver')" :description="order.deliveryTime ? formatDateTime(order.deliveryTime) : ''"></el-step>
        <el-step :title="t('oms.order.detail.stepConfirmReceive')" :description="order.receiveTime ? formatDateTime(order.receiveTime) : ''"></el-step>
        <el-step :title="t('oms.order.detail.stepCompleteReview')" :description="order.commentTime ? formatDateTime(order.commentTime) : ''"></el-step>
      </el-steps>
    </div>
    <el-card shadow="never" style="margin-top: 15px">
      <div class="operate-container">
        <el-icon class="color-danger el-icon-middle" style="margin-left: 15px;">
          <Warning />
        </el-icon>
        <span class="color-danger">{{ t('oms.order.detail.currentStatus') }}：{{ formatStatus(order.status) }}</span>
        <div class="operate-button-container" v-show="order.status === 0">
          <el-button size="small" @click="showUpdateReceiverDialog">{{ t('oms.order.detail.btnModifyReceiver') }}</el-button>
          <el-button size="small" @click="showUpdateMoneyDialog">{{ t('oms.order.detail.btnModifyMoney') }}</el-button>
          <el-button size="small" @click="showMessageDialog">{{ t('oms.order.detail.btnSendMessage') }}</el-button>
          <el-button size="small" @click="showCloseOrderDialog">{{ t('oms.order.detail.btnCloseOrder') }}</el-button>
          <el-button size="small" @click="showMarkOrderDialog">{{ t('oms.order.detail.btnMarkOrder') }}</el-button>
        </div>
        <div class="operate-button-container" v-show="order.status === 1">
          <el-button size="small" @click="showUpdateReceiverDialog">{{ t('oms.order.detail.btnModifyReceiver') }}</el-button>
          <el-button size="small" @click="showMessageDialog">{{ t('oms.order.detail.btnSendMessage') }}</el-button>
          <el-button size="small">{{ t('oms.order.detail.btnCancelOrder') }}</el-button>
          <el-button size="small" @click="showMarkOrderDialog">{{ t('oms.order.detail.btnMarkOrder') }}</el-button>
        </div>
        <div class="operate-button-container" v-show="order.status === 2 || order.status === 3">
          <el-button size="small" @click="showLogisticsDialog">{{ t('oms.order.detail.btnOrderTrack') }}</el-button>
          <el-button size="small" @click="showMessageDialog">{{ t('oms.order.detail.btnSendMessage') }}</el-button>
          <el-button size="small" @click="showMarkOrderDialog">{{ t('oms.order.detail.btnMarkOrder') }}</el-button>
        </div>
        <div class="operate-button-container" v-show="order.status === 4">
          <el-button size="small" @click="handleDeleteOrder">{{ t('oms.order.detail.btnDeleteOrder') }}</el-button>
          <el-button size="small" @click="showMarkOrderDialog">{{ t('oms.order.detail.btnMarkOrder') }}</el-button>
        </div>
      </div>
      <!-- The rest of the template is unchanged; most template code is omitted for brevity -->
      <!-- Refer to the original file for the complete template -->
      <div style="margin-top: 20px">
        <svg-icon icon-class="marker" style="color: #606266"></svg-icon>
        <span class="font-small">{{ t('oms.detail.basicInfo') }}</span>
      </div>
      <div class="table-layout">
        <el-row>
          <el-col :span="4" class="table-cell-title">{{ t('oms.order.list.orderSn') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.deliverySn') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.order.list.userAccount') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.order.list.payType') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.order.list.sourceType') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.orderType') }}</el-col>
        </el-row>
        <el-row>
          <el-col :span="4" class="table-cell">{{ order.orderSn }}</el-col>
          <el-col :span="4" class="table-cell">{{ t('oms.detail.cellNoData') }}</el-col>
          <el-col :span="4" class="table-cell">{{ order.memberUsername }}</el-col>
          <el-col :span="4" class="table-cell">{{ formatPayType(order.payType) }}</el-col>
          <el-col :span="4" class="table-cell">{{ formatSourceType(order.sourceType) }}</el-col>
          <el-col :span="4" class="table-cell">{{ formatOrderType(order.orderType) }}</el-col>
        </el-row>
        <el-row>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.deliveryMethod') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.deliverySnTitle') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.autoConfirmDay') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.integration') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.growth') }}</el-col>
          <el-col :span="4" class="table-cell-title">{{ t('oms.detail.promotionInfo') }}</el-col>
        </el-row>
        <el-row>
          <el-col :span="4" class="table-cell">{{ formatNull(order.deliveryCompany) }}</el-col>
          <el-col :span="4" class="table-cell">{{ formatNull(order.deliverySn) }}</el-col>
          <el-col :span="4" class="table-cell">{{ order.autoConfirmDay }}{{ t('oms.orderSetting.days') }}</el-col>
          <el-col :span="4" class="table-cell">{{ order.integration }}</el-col>
          <el-col :span="4" class="table-cell">{{ order.growth }}</el-col>
          <el-col :span="4" class="table-cell">
            <el-popover placement="top-start" :title="t('oms.detail.promotionInfo')" width="200" trigger="hover" :content="order.promotionInfo">
              <template v-slot:reference>
                <span>{{ formatLongText(order.promotionInfo) }}</span>
              </template>
            </el-popover>
          </el-col>
        </el-row>
      </div>
      <div style="margin-top: 20px">
        <svg-icon icon-class="marker" style="color: #606266"></svg-icon>
        <span class="font-small">收货人信息</span>
      </div>
      <div class="table-layout">
        <el-row>
          <el-col :span="6" class="table-cell-title">收货人</el-col>
          <el-col :span="6" class="table-cell-title">手机号码</el-col>
          <el-col :span="6" class="table-cell-title">邮政编码</el-col>
          <el-col :span="6" class="table-cell-title">收货地址</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell">{{ order.receiverName }}</el-col>
          <el-col :span="6" class="table-cell">{{ order.receiverPhone }}</el-col>
          <el-col :span="6" class="table-cell">{{ order.receiverPostCode }}</el-col>
          <el-col :span="6" class="table-cell">{{ formatAddress(order) }}</el-col>
        </el-row>
      </div>
      <div style="margin-top: 20px">
        <svg-icon icon-class="marker" style="color: #606266"></svg-icon>
        <span class="font-small">商品信息</span>
      </div>
      <el-table ref="orderItemTable" :data="order.orderItemList" style="width: 100%;margin-top: 20px" border>
        <el-table-column label="商品图片" width="120" align="center">
          <template v-slot="scope">
            <img :src="scope.row.productPic" style="height: 80px">
          </template>
        </el-table-column>
        <el-table-column label="商品名称" align="center">
          <template v-slot="scope">
            <p>{{ scope.row.productName }}</p>
            <p>{{ t('product.brand') }}{{ scope.row.productBrand }}</p>
          </template>
        </el-table-column>
        <el-table-column label="价格/货号" width="160" align="center">
          <template v-slot="scope">
            <p>{{ t('oms.price') }}{{ scope.row.productPrice }}</p>
            <p>{{ t('product.sn') }}{{ scope.row.productSn }}</p>
          </template>
        </el-table-column>
        <el-table-column label="属性" width="160" align="center">
          <template v-slot="scope">
            {{ formatProductAttr(scope.row.productAttr) }}
          </template>
        </el-table-column>
        <el-table-column label="数量" width="120" align="center">
          <template v-slot="scope">
            {{ scope.row.productQuantity }}
          </template>
        </el-table-column>
        <el-table-column label="小计" width="120" align="center">
          <template v-slot="scope">
            ￥{{ scope.row.productPrice * scope.row.productQuantity }}
          </template>
        </el-table-column>
      </el-table>
      <div style="float: right;margin: 20px">
        合计：<span class="color-danger">￥{{ order.totalAmount }}</span>
      </div>
      <div style="margin-top: 60px">
        <svg-icon icon-class="marker" style="color: #606266"></svg-icon>
        <span class="font-small">费用信息</span>
      </div>
      <div class="table-layout">
        <el-row>
          <el-col :span="6" class="table-cell-title">商品合计</el-col>
          <el-col :span="6" class="table-cell-title">运费</el-col>
          <el-col :span="6" class="table-cell-title">优惠券</el-col>
          <el-col :span="6" class="table-cell-title">积分抵扣</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell">￥{{ order.totalAmount }}</el-col>
          <el-col :span="6" class="table-cell">￥{{ order.freightAmount }}</el-col>
          <el-col :span="6" class="table-cell">-￥{{ order.couponAmount }}</el-col>
          <el-col :span="6" class="table-cell">-￥{{ order.integrationAmount }}</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell-title">活动优惠</el-col>
          <el-col :span="6" class="table-cell-title">折扣金额</el-col>
          <el-col :span="6" class="table-cell-title">订单总金额</el-col>
          <el-col :span="6" class="table-cell-title">应付款金额</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell">-￥{{ order.promotionAmount }}</el-col>
          <el-col :span="6" class="table-cell">-￥{{ order.discountAmount }}</el-col>
          <el-col :span="6" class="table-cell">
            <span class="color-danger">￥{{ order.totalAmount + order.freightAmount }}</span>
          </el-col>
          <el-col :span="6" class="table-cell">
            <span class="color-danger">￥{{ order.payAmount + order.freightAmount - order.discountAmount }}</span>
          </el-col>
        </el-row>
      </div>
      <div style="margin-top: 20px">
        <svg-icon icon-class="marker" style="color: #606266"></svg-icon>
        <span class="font-small">操作信息</span>
      </div>
      <el-table style="margin-top: 20px;width: 100%" ref="orderHistoryTable" :data="order.historyList" border>
        <el-table-column label="操作者" width="120" align="center">
          <template v-slot="scope">
            {{ scope.row.operateMan }}
          </template>
        </el-table-column>
        <el-table-column label="操作时间" width="160" align="center">
          <template v-slot="scope">
            {{ formatDateTime(scope.row.createTime) }}
          </template>
        </el-table-column>
        <el-table-column label="订单状态" width="120" align="center">
          <template v-slot="scope">
            {{ formatStatus(scope.row.orderStatus) }}
          </template>
        </el-table-column>
        <el-table-column label="付款状态" width="120" align="center">
          <template v-slot="scope">
            {{ formatPayStatus(scope.row.orderStatus) }}
          </template>
        </el-table-column>
        <el-table-column label="发货状态" width="120" align="center">
          <template v-slot="scope">
            {{ formatDeliverStatus(scope.row.orderStatus) }}
          </template>
        </el-table-column>
        <el-table-column label="备注" align="center">
          <template v-slot="scope">
            {{ scope.row.note }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    <el-dialog title="修改收货人信息" v-model="receiverDialogVisible" width="40%">
      <el-form :model="receiverInfo" ref="receiverInfoForm" label-width="150px">
        <el-form-item label="收货人姓名：">
          <el-input v-model="receiverInfo.receiverName" style="width: 200px"></el-input>
        </el-form-item>
        <el-form-item label="手机号码：">
          <el-input v-model="receiverInfo.receiverPhone" style="width: 200px">
          </el-input>
        </el-form-item>
        <el-form-item label="邮政编码：">
          <el-input v-model="receiverInfo.receiverPostCode" style="width: 200px">
          </el-input>
        </el-form-item>
        <el-form-item label="所在区域：">
          <el-cascader v-model="selectedRegions" :options="(pcaTextArr as CascaderOption[])"
            @change="onSelectRegionChange" placeholder="请选择省市区"></el-cascader>
        </el-form-item>
        <el-form-item label="详细地址：">
          <el-input v-model="receiverInfo.receiverDetailAddress" type="textarea" :rows="3">
          </el-input>
        </el-form-item>
      </el-form>
      <template v-slot:footer>
        <span class="dialog-footer">
          <el-button @click="receiverDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleUpdateReceiverInfo">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
    <el-dialog title="修改费用信息" v-model="moneyDialogVisible" width="40%">
      <div class="table-layout">
        <el-row>
          <el-col :span="6" class="table-cell-title">商品合计</el-col>
          <el-col :span="6" class="table-cell-title">运费</el-col>
          <el-col :span="6" class="table-cell-title">优惠券</el-col>
          <el-col :span="6" class="table-cell-title">积分抵扣</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell">￥{{ order.totalAmount }}</el-col>
          <el-col :span="6" class="table-cell">
            <el-input v-model.number="moneyInfo.freightAmount" size="small"><template
                v-slot:prepend>￥</template></el-input>
          </el-col>
          <el-col :span="6" class="table-cell">-￥{{ order.couponAmount }}</el-col>
          <el-col :span="6" class="table-cell">-￥{{ order.integrationAmount }}</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell-title">活动优惠</el-col>
          <el-col :span="6" class="table-cell-title">折扣金额</el-col>
          <el-col :span="6" class="table-cell-title">订单总金额</el-col>
          <el-col :span="6" class="table-cell-title">应付款金额</el-col>
        </el-row>
        <el-row>
          <el-col :span="6" class="table-cell">-￥{{ order.promotionAmount }}</el-col>
          <el-col :span="6" class="table-cell">
            <el-input v-model.number="moneyInfo.discountAmount" size="small"><template
                v-slot:prepend>-￥</template></el-input>
          </el-col>
          <el-col :span="6" class="table-cell">
            <span class="color-danger">￥{{ order.totalAmount + moneyInfo.freightAmount }}</span>
          </el-col>
          <el-col :span="6" class="table-cell">
            <span class="color-danger">￥{{ order.payAmount + moneyInfo.freightAmount - moneyInfo.discountAmount
            }}</span>
          </el-col>
        </el-row>
      </div>
      <template v-slot:footer>
        <span class="dialog-footer">
          <el-button @click="moneyDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleUpdateMoneyInfo">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
    <el-dialog title="发送站内信" v-model="messageDialogVisible" width="40%">
      <el-form :model="message" ref="receiverInfoForm" label-width="150px">
        <el-form-item label="标题：">
          <el-input v-model="message.title" style="width: 200px"></el-input>
        </el-form-item>
        <el-form-item label="内容：">
          <el-input v-model="message.content" type="textarea" :rows="3">
          </el-input>
        </el-form-item>
      </el-form>
      <template v-slot:footer>
        <span class="dialog-footer">
          <el-button @click="messageDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleSendMessage">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
    <el-dialog title="关闭订单" v-model:visible="closeDialogVisible" width="40%">
      <el-form :model="closeInfo" label-width="150px">
        <el-form-item label="操作备注：">
          <el-input v-model="closeInfo.note" type="textarea" :rows="3">
          </el-input>
        </el-form-item>
      </el-form>
      <template v-slot:footer>
        <span class="dialog-footer">
          <el-button @click="closeDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleCloseOrder">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
    <el-dialog title="备注订单" v-model="markOrderDialogVisible" width="40%">
      <el-form :model="markInfo" label-width="150px">
        <el-form-item label="操作备注：">
          <el-input v-model="markInfo.note" type="textarea" :rows="3">
          </el-input>
        </el-form-item>
      </el-form>
      <template v-slot:footer>
        <span class="dialog-footer">
          <el-button @click="markOrderDialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleMarkOrder">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
    <logistics-dialog v-model="logisticsDialogVisible"></logistics-dialog>
  </div>
</template>

<style scoped>
.detail-container {
  width: 80%;
  padding: 20px 20px 20px 20px;
  margin: 20px auto;
}

.operate-container {
  background: #F2F6FC;
  height: 80px;
  margin: -20px -20px 0;
  line-height: 80px;
}

.operate-button-container {
  float: right;
  margin-right: 20px
}

.table-layout {
  margin-top: 20px;
  border-left: 1px solid #DCDFE6;
  border-top: 1px solid #DCDFE6;
}

.table-cell {
  height: 60px;
  line-height: 40px;
  border-right: 1px solid #DCDFE6;
  border-bottom: 1px solid #DCDFE6;
  padding: 10px;
  font-size: 14px;
  color: #606266;
  text-align: center;
  overflow: hidden;
}

.table-cell-title {
  border-right: 1px solid #DCDFE6;
  border-bottom: 1px solid #DCDFE6;
  padding: 10px;
  background: #F2F6FC;
  text-align: center;
  font-size: 14px;
  color: #303133;
}
</style>
