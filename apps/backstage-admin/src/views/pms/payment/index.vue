<script setup lang=\"ts\">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh, View, Money } from '@element-plus/icons-vue'
import {
  getPaymentOrderListAPI,
  type PaymentOrder,
  type PaymentQueryParam,
  verifyPaymentAPI,
  queryPaymentStatusAPI
} from '@/apis/payment'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<PaymentQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<PaymentOrder[]>([])
const listLoading = ref(true)
const total = ref(0)

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getPaymentOrderListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data.list || []
    total.value = response.data.total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error(t('payment.order.list.loadFailed'))
  }
}

onMounted(() => {
  getList()
})

const multipleSelection = ref<PaymentOrder[]>([])

const handleSelectionChange = (val: PaymentOrder[]) => {
  multipleSelection.value = val
}

const handleSearch = () => {
  listQuery.value.pageNum = 1
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageSize = val
  getList()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

const handleVerify = async (row: PaymentOrder) => {
  try {
    await ElMessageBox.confirm(
      t('payment.order.verify.confirm'),
      t('common.verify'),
      {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
      }
    )
    const response = await verifyPaymentAPI(row.orderNo, row.transactionId || '')
    if (response.data) {
      ElMessage.success(t('payment.order.verify.success'))
      getList()
    }
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('payment.order.verify.failed'))
    }
  }
}

const handleQueryStatus = async (row: PaymentOrder) => {
  try {
    const response = await queryPaymentStatusAPI(row.orderNo)
    ElMessage.success(t('payment.order.status.querySuccess') + ': ' + JSON.stringify(response.data))
  } catch (error) {
    ElMessage.error(t('payment.order.status.queryFailed'))
  }
}

const statusOptions = [
  { label: t('payment.order.status.pending'), value: 0 },
  { label: t('payment.order.status.processing'), value: 1 },
  { label: t('payment.order.status.success'), value: 2 },
  { label: t('payment.order.status.failed'), value: 3 },
  { label: t('payment.order.status.refunded'), value: 4 }
]

const paymentMethodOptions = [
  { label: 'WeChat Pay', value: 'wechat' },
  { label: 'Alipay', value: 'alipay' },
  { label: 'Stripe', value: 'stripe' },
  { label: 'Crypto', value: 'crypto' }
]

const getStatusLabel = (status: number) => {
  const option = statusOptions.find(opt => opt.value === status)
  return option ? option.label : '-'
}

const getMethodLabel = (method: string) => {
  const option = paymentMethodOptions.find(opt => opt.value === method)
  return option ? option.label : method
}
</script>

<template>
  <div class=\"app-container\">
    <el-card class=\"search-card\" shadow=\"never\">
      <div class=\"search-form\">
        <el-form :inline=\"true\" :model=\"listQuery\" class=\"demo-form-inline\">
          <el-form-item :label=\"t('payment.order.list.orderNo')\">
            <el-input
              v-model=\"listQuery.orderNo\"
              :placeholder=\"t('payment.order.list.orderNoPlaceholder')\"
              clearable
              @keyup.enter=\"handleSearch\"
            />
          </el-form-item>
          <el-form-item :label=\"t('payment.order.list.userId')\">
            <el-input
              v-model=\"listQuery.userId\"
              :placeholder=\"t('payment.order.list.userIdPlaceholder')\"
              clearable
              type=\"number\"
              @keyup.enter=\"handleSearch\"
            />
          </el-form-item>
          <el-form-item :label=\"t('payment.order.list.status')\">
            <el-select v-model=\"listQuery.status\" :placeholder=\"t('common.select')\" clearable>
              <el-option
                v-for=\"item in statusOptions\"
                :key=\"item.value\"
                :label=\"item.label\"
                :value=\"item.value\"
              />
            </el-select>
          </el-form-item>
          <el-form-item :label=\"t('payment.order.list.paymentMethod')\">
            <el-select v-model=\"listQuery.paymentMethod\" :placeholder=\"t('common.select')\" clearable>
              <el-option
                v-for=\"item in paymentMethodOptions\"
                :key=\"item.value\"
                :label=\"item.label\"
                :value=\"item.value\"
              />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type=\"primary\" :icon=\"Search\" @click=\"handleSearch\">
              {{ t('common.search') }}
            </el-button>
            <el-button :icon=\"Refresh\" @click=\"handleReset\">
              {{ t('common.reset') }}
            </el-button>
          </el-form-item>
        </el-form>
      </div>
    </el-card>

    <el-card class=\"table-card\" shadow=\"never\">
      <template #header>
        <div class=\"card-header\">
          <span>{{ t('payment.order.list.title') }}</span>
        </div>
      </template>

      <el-table
        v-loading=\"listLoading\"
        :data=\"list\"
        border
        stripe
        @selection-change=\"handleSelectionChange\"
      >
        <el-table-column type=\"selection\" width=\"55\" align=\"center\" />
        <el-table-column label=\"ID\" prop=\"id\" width=\"80\" align=\"center\" />
        <el-table-column :label=\"t('payment.order.list.orderNo')\" prop=\"orderNo\" min-width=\"180\" />
        <el-table-column :label=\"t('payment.order.list.userId')\" prop=\"userId\" width=\"100\" align=\"center\" />
        <el-table-column :label=\"t('payment.order.list.amount')\" prop=\"amount\" width=\"120\" align=\"right\">
          <template #default=\"{ row }\">
            {{ row.currency }} {{ row.amount?.toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column :label=\"t('payment.order.list.paymentMethod')\" prop=\"paymentMethod\" width=\"120\" align=\"center\">
          <template #default=\"{ row }\">
            {{ getMethodLabel(row.paymentMethod) }}
          </template>
        </el-table-column>
        <el-table-column :label=\"t('payment.order.list.status')\" prop=\"status\" width=\"120\" align=\"center\">
          <template #default=\"{ row }\">
            <el-tag :type=\"row.status === 2 ? 'success' : row.status === 3 ? 'danger' : 'warning'\">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label=\"t('payment.order.list.transactionId')\" prop=\"transactionId\" min-width=\"180\" />
        <el-table-column :label=\"t('payment.order.list.createdAt')\" prop=\"createdAt\" width=\"160\">
          <template #default=\"{ row }\">
            {{ formatDateTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column :label=\"t('common.action')\" width=\"180\" align=\"center\" fixed=\"right\">
          <template #default=\"{ row }\">
            <el-button link type=\"primary\" :icon=\"View\" @click=\"handleVerify(row)\">
              {{ t('payment.order.verify.button') }}
            </el-button>
            <el-button link type=\"primary\" :icon=\"Money\" @click=\"handleQueryStatus(row)\">
              {{ t('payment.order.status.button') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class=\"pagination-container\">
        <el-pagination
          v-model:current-page=\"listQuery.pageNum\"
          v-model:page-size=\"listQuery.pageSize\"
          :page-sizes=\"[10, 20, 50, 100]\"
          :total=\"total\"
          layout=\"total, sizes, prev, pager, next, jumper\"
          @size-change=\"handleSizeChange\"
          @current-change=\"handleCurrentChange\"
        />
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.search-card {
  margin-bottom: 20px;
}

.search-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.table-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
</style>