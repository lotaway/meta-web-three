<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh } from '@element-plus/icons-vue'
import {
  getKYCListAPI,
  reviewKYCAPI,
  getKYCStatisticsAPI,
  getCryptoPriceListAPI,
  getLatestCryptoPricesAPI,
  getExchangeOrderListAPI,
  getExchangeOrderStatisticsAPI,
  getPaymentStatisticsAPI,
  type UserKYC,
  type UserKYCQueryParams,
  type CryptoPrice,
  type CryptoPriceQueryParams,
  type ExchangeOrder,
  type ExchangeOrderQueryParams,
  type KYCStatistics,
  type ExchangeOrderStatistics,
  type PaymentStatistics
} from '@/apis/payment'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

// State
const loading = ref(false)
const activeTab = ref('kyc')
const statistics = ref<PaymentStatistics>({
  totalKYC: 0,
  pendingKYC: 0,
  totalOrders: 0,
  completedOrders: 0,
  cryptoSymbols: 0
})

// KYC Query params
const kycQuery = reactive<UserKYCQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  userId: undefined,
  level: undefined,
  status: undefined
})

const kycList = ref<UserKYC[]>([])
const kycTotal = ref(0)

// Crypto Price Query params
const cryptoQuery = reactive<CryptoPriceQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  symbol: '',
  baseCurrency: '',
  quoteCurrency: '',
  source: ''
})

const cryptoList = ref<CryptoPrice[]>([])
const cryptoTotal = ref(0)

// Exchange Order Query params
const orderQuery = reactive<ExchangeOrderQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  userId: undefined,
  orderNo: '',
  status: undefined,
  orderType: undefined
})

const orderList = ref<ExchangeOrder[]>([])
const orderTotal = ref(0)

// Review dialog state
const reviewDialogVisible = ref(false)
const reviewForm = reactive({
  status: 'APPROVED',
  reviewerId: '',
  reviewNotes: ''
})
const reviewingId = ref<number | undefined>()

// Options
const kycLevelOptions = [
  { label: 'L0 - Basic', value: 'L0' },
  { label: 'L1 - Identity', value: 'L1' },
  { label: 'L2 - Advanced', value: 'L2' },
  { label: 'L3 - Enterprise', value: 'L3' }
]

const kycStatusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'Approved', value: 'APPROVED' },
  { label: 'Rejected', value: 'REJECTED' },
  { label: 'Expired', value: 'EXPIRED' }
]

const orderStatusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'Paid', value: 'PAID' },
  { label: 'Processing', value: 'PROCESSING' },
  { label: 'Completed', value: 'COMPLETED' },
  { label: 'Failed', value: 'FAILED' },
  { label: 'Expired', value: 'EXPIRED' },
  { label: 'Cancelled', value: 'CANCELLED' }
]

const orderTypeOptions = [
  { label: 'Buy Crypto', value: 'BUY_CRYPTO' },
  { label: 'Sell Crypto', value: 'SELL_CRYPTO' }
]

const getKYCStatusTagType = (status: string): 'success' | 'info' | 'danger' | 'warning' => {
  const map: Record<string, 'success' | 'info' | 'danger' | 'warning'> = {
    'PENDING': 'warning',
    'APPROVED': 'success',
    'REJECTED': 'danger',
    'EXPIRED': 'info'
  }
  return map[status] || 'info'
}

const getOrderStatusTagType = (status: string): 'success' | 'info' | 'danger' | 'warning' => {
  const map: Record<string, 'success' | 'info' | 'danger' | 'warning'> = {
    'PENDING': 'warning',
    'PAID': 'warning',
    'PROCESSING': 'warning',
    'COMPLETED': 'success',
    'FAILED': 'danger',
    'EXPIRED': 'info',
    'CANCELLED': 'info'
  }
  return map[status] || 'info'
}

const getOrderTypeTagType = (type: string): 'success' | 'info' | 'danger' | 'warning' => {
  return type === 'BUY_CRYPTO' ? 'success' : 'warning'
}

// Load statistics
const loadStatistics = async () => {
  try {
    const res = await getPaymentStatisticsAPI()
    statistics.value = res.data
  } catch (error) {
    console.error('Failed to load statistics:', error)
  }
}

// Load KYC list
const getKYCList = async () => {
  loading.value = true
  try {
    const res = await getKYCListAPI(kycQuery)
    kycList.value = res.data.list
    kycTotal.value = res.data.total
  } catch (error) {
    console.error('Failed to load KYC list:', error)
  } finally {
    loading.value = false
  }
}

// Load crypto price list
const getCryptoList = async () => {
  loading.value = true
  try {
    const res = await getCryptoPriceListAPI(cryptoQuery)
    cryptoList.value = res.data.list
    cryptoTotal.value = res.data.total
  } catch (error) {
    console.error('Failed to load crypto price list:', error)
  } finally {
    loading.value = false
  }
}

// Load exchange order list
const getOrderList = async () => {
  loading.value = true
  try {
    const res = await getExchangeOrderListAPI(orderQuery)
    orderList.value = res.data.list
    orderTotal.value = res.data.total
  } catch (error) {
    console.error('Failed to load exchange order list:', error)
  } finally {
    loading.value = false
  }
}

// Handle KYC search
const handleKYCSearch = () => {
  kycQuery.pageNum = 1
  getKYCList()
}

// Handle KYC reset
const handleKYCReset = () => {
  kycQuery.userId = undefined
  kycQuery.level = undefined
  kycQuery.status = undefined
  kycQuery.pageNum = 1
  getKYCList()
}

// Handle crypto search
const handleCryptoSearch = () => {
  cryptoQuery.pageNum = 1
  getCryptoList()
}

// Handle crypto reset
const handleCryptoReset = () => {
  cryptoQuery.symbol = ''
  cryptoQuery.baseCurrency = ''
  cryptoQuery.quoteCurrency = ''
  cryptoQuery.source = ''
  cryptoQuery.pageNum = 1
  getCryptoList()
}

// Handle order search
const handleOrderSearch = () => {
  orderQuery.pageNum = 1
  getOrderList()
}

// Handle order reset
const handleOrderReset = () => {
  orderQuery.userId = undefined
  orderQuery.orderNo = ''
  orderQuery.status = undefined
  orderQuery.orderType = undefined
  orderQuery.pageNum = 1
  getOrderList()
}

const handleViewKYC = (row: UserKYC) => {
  // View KYC details - could open a dialog
  ElMessage.info(`View KYC ID: ${row.id}`)
}

// Handle review KYC
const handleReviewKYC = (row: UserKYC) => {
  reviewingId.value = row.id
  reviewForm.status = 'APPROVED'
  reviewForm.reviewerId = ''
  reviewForm.reviewNotes = ''
  reviewDialogVisible.value = true
}

// Submit review
const handleReviewSubmit = async () => {
  try {
    await reviewKYCAPI(reviewingId.value!, {
      status: reviewForm.status,
      reviewerId: reviewForm.reviewerId,
      reviewNotes: reviewForm.reviewNotes
    })
    ElMessage.success({ message: t('common.updateSuccess'), duration: MESSAGE_DURATION_SHORT })
    reviewDialogVisible.value = false
    getKYCList()
    loadStatistics()
  } catch (error) {
    console.error('Failed to review KYC:', error)
  }
}

// Handle tab change
const handleTabChange = (tab: any) => {
  activeTab.value = tab.props.name
  if (activeTab.value === 'kyc') {
    getKYCList()
  } else if (activeTab.value === 'crypto') {
    getCryptoList()
  } else if (activeTab.value === 'orders') {
    getOrderList()
  }
}

// Handle pagination
const handleKYCPageChange = (page: number) => {
  kycQuery.pageNum = page
  getKYCList()
}

const handleCryptoPageChange = (page: number) => {
  cryptoQuery.pageNum = page
  getCryptoList()
}

const handleOrderPageChange = (page: number) => {
  orderQuery.pageNum = page
  getOrderList()
}

// Format functions
const formatDateTime = (dateTime: string) => {
  if (!dateTime) return '-'
  return dateTime.slice(0, 19).replace('T', ' ')
}

const formatAmount = (amount: number, decimals = 2) => {
  if (amount === undefined || amount === null) return '-'
  return amount.toFixed(decimals)
}

// Initial load
onMounted(() => {
  loadStatistics()
  getKYCList()
})
</script>

<template>
  <div class="payment-container">
    <el-card class="statistics-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('payment.overview') }}</span>
        </div>
      </template>
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">{{ t('payment.totalKYC') }}</div>
            <div class="stat-value">{{ statistics.totalKYC }}</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">{{ t('payment.pendingKYC') }}</div>
            <div class="stat-value warning">{{ statistics.pendingKYC }}</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">{{ t('payment.totalOrders') }}</div>
            <div class="stat-value">{{ statistics.totalOrders }}</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">{{ t('payment.completedOrders') }}</div>
            <div class="stat-value success">{{ statistics.completedOrders }}</div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <el-card class="table-card">
      <el-tabs v-model="activeTab" @tab-change="handleTabChange">
        <el-tab-pane :label="t('payment.kycManagement')" name="kyc">
          <div class="toolbar">
            <el-form :inline="true" :model="kycQuery">
              <el-form-item :label="t('payment.userId')">
                <el-input v-model="kycQuery.userId" :placeholder="t('payment.userId')" clearable style="width: 120px" />
              </el-form-item>
              <el-form-item :label="t('payment.level')">
                <el-select v-model="kycQuery.level" :placeholder="t('payment.level')" clearable style="width: 150px">
                  <el-option v-for="item in kycLevelOptions" :key="item.value" :label="item.label" :value="item.value" />
                </el-select>
              </el-form-item>
              <el-form-item :label="t('payment.status')">
                <el-select v-model="kycQuery.status" :placeholder="t('payment.status')" clearable style="width: 120px">
                  <el-option v-for="item in kycStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
                </el-select>
              </el-form-item>
              <el-form-item>
                <el-button type="primary" :icon="Search" @click="handleKYCSearch">{{ t('common.search') }}</el-button>
                <el-button :icon="Refresh" @click="handleKYCReset">{{ t('common.reset') }}</el-button>
              </el-form-item>
            </el-form>
          </div>

          <el-table :data="kycList" v-loading="loading" stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="userId" :label="t('payment.userId')" width="100" />
            <el-table-column prop="realName" :label="t('payment.realName')" width="120" />
            <el-table-column prop="level" :label="t('payment.level')" width="100" />
            <el-table-column :label="t('payment.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="getKYCStatusTagType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="country" :label="t('payment.country')" width="100" />
            <el-table-column prop="submittedAt" :label="t('payment.submittedAt')" width="160">
              <template #default="{ row }">
                {{ formatDateTime(row.submittedAt) }}
              </template>
            </el-table-column>
            <el-table-column prop="reviewedAt" :label="t('payment.reviewedAt')" width="160">
              <template #default="{ row }">
                {{ formatDateTime(row.reviewedAt) }}
              </template>
            </el-table-column>
            <el-table-column :label="t('common.actions')" width="150" fixed="right">
              <template #default="{ row }">
                <el-button v-if="row.status === 'PENDING'" type="primary" size="small" @click="handleReviewKYC(row)">
                  {{ t('payment.review') }}
                </el-button>
                <el-button type="info" size="small" @click="handleViewKYC(row)">
                  {{ t('common.view') }}
                </el-button>
              </template>
            </el-table-column>
          </el-table>

          <div class="pagination">
            <el-pagination
              :current-page="kycQuery.pageNum"
              :page-size="kycQuery.pageSize"
              :page-size-options="PAGE_SIZE_OPTIONS"
              :total="kycTotal"
              layout="total, sizes, prev, pager, next"
              @size-change="(size: number) => { kycQuery.pageSize = size; getKYCList() }"
              @current-change="handleKYCPageChange"
            />
          </div>
        </el-tab-pane>

        <el-tab-pane :label="t('payment.cryptoPrice')" name="crypto">
          <div class="toolbar">
            <el-form :inline="true" :model="cryptoQuery">
              <el-form-item :label="t('payment.symbol')">
                <el-input v-model="cryptoQuery.symbol" :placeholder="t('payment.symbol')" clearable style="width: 120px" />
              </el-form-item>
              <el-form-item :label="t('payment.baseCurrency')">
                <el-input v-model="cryptoQuery.baseCurrency" :placeholder="t('payment.baseCurrency')" clearable style="width: 100px" />
              </el-form-item>
              <el-form-item :label="t('payment.quoteCurrency')">
                <el-input v-model="cryptoQuery.quoteCurrency" :placeholder="t('payment.quoteCurrency')" clearable style="width: 100px" />
              </el-form-item>
              <el-form-item :label="t('payment.source')">
                <el-input v-model="cryptoQuery.source" :placeholder="t('payment.source')" clearable style="width: 100px" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" :icon="Search" @click="handleCryptoSearch">{{ t('common.search') }}</el-button>
                <el-button :icon="Refresh" @click="handleCryptoReset">{{ t('common.reset') }}</el-button>
              </el-form-item>
            </el-form>
          </div>

          <el-table :data="cryptoList" v-loading="loading" stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="symbol" :label="t('payment.symbol')" width="120" />
            <el-table-column prop="baseCurrency" :label="t('payment.baseCurrency')" width="100" />
            <el-table-column prop="quoteCurrency" :label="t('payment.quoteCurrency')" width="100" />
            <el-table-column prop="price" :label="t('payment.price')" width="150">
              <template #default="{ row }">
                {{ formatAmount(row.price) }}
              </template>
            </el-table-column>
            <el-table-column prop="bidPrice" :label="t('payment.bidPrice')" width="120">
              <template #default="{ row }">
                {{ formatAmount(row.bidPrice) }}
              </template>
            </el-table-column>
            <el-table-column prop="askPrice" :label="t('payment.askPrice')" width="120">
              <template #default="{ row }">
                {{ formatAmount(row.askPrice) }}
              </template>
            </el-table-column>
            <el-table-column prop="changePercent24h" :label="t('payment.change24h')" width="100">
              <template #default="{ row }">
                <span :class="row.changePercent24h >= 0 ? 'positive' : 'negative'">
                  {{ formatAmount(row.changePercent24h) }}%
                </span>
              </template>
            </el-table-column>
            <el-table-column prop="source" :label="t('payment.source')" width="100" />
            <el-table-column prop="timestamp" :label="t('payment.timestamp')" width="160">
              <template #default="{ row }">
                {{ formatDateTime(row.timestamp) }}
              </template>
            </el-table-column>
          </el-table>

          <div class="pagination">
            <el-pagination
              :current-page="cryptoQuery.pageNum"
              :page-size="cryptoQuery.pageSize"
              :page-size-options="PAGE_SIZE_OPTIONS"
              :total="cryptoTotal"
              layout="total, sizes, prev, pager, next"
              @size-change="(size: number) => { cryptoQuery.pageSize = size; getCryptoList() }"
              @current-change="handleCryptoPageChange"
            />
          </div>
        </el-tab-pane>

        <el-tab-pane :label="t('payment.exchangeOrders')" name="orders">
          <div class="toolbar">
            <el-form :inline="true" :model="orderQuery">
              <el-form-item :label="t('payment.userId')">
                <el-input v-model="orderQuery.userId" :placeholder="t('payment.userId')" clearable style="width: 120px" />
              </el-form-item>
              <el-form-item :label="t('payment.orderNo')">
                <el-input v-model="orderQuery.orderNo" :placeholder="t('payment.orderNo')" clearable style="width: 150px" />
              </el-form-item>
              <el-form-item :label="t('payment.orderType')">
                <el-select v-model="orderQuery.orderType" :placeholder="t('payment.orderType')" clearable style="width: 130px">
                  <el-option v-for="item in orderTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
                </el-select>
              </el-form-item>
              <el-form-item :label="t('payment.status')">
                <el-select v-model="orderQuery.status" :placeholder="t('payment.status')" clearable style="width: 120px">
                  <el-option v-for="item in orderStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
                </el-select>
              </el-form-item>
              <el-form-item>
                <el-button type="primary" :icon="Search" @click="handleOrderSearch">{{ t('common.search') }}</el-button>
                <el-button :icon="Refresh" @click="handleOrderReset">{{ t('common.reset') }}</el-button>
              </el-form-item>
            </el-form>
          </div>

          <el-table :data="orderList" v-loading="loading" stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="orderNo" :label="t('payment.orderNo')" width="180" />
            <el-table-column prop="userId" :label="t('payment.userId')" width="100" />
            <el-table-column :label="t('payment.orderType')" width="120">
              <template #default="{ row }">
                <el-tag :type="getOrderTypeTagType(row.orderType)">{{ row.orderType }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column :label="t('payment.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="getOrderStatusTagType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="fiatCurrency" :label="t('payment.fiatCurrency')" width="100" />
            <el-table-column prop="fiatAmount" :label="t('payment.fiatAmount')" width="120">
              <template #default="{ row }">
                {{ formatAmount(row.fiatAmount) }}
              </template>
            </el-table-column>
            <el-table-column prop="cryptoCurrency" :label="t('payment.cryptoCurrency')" width="120" />
            <el-table-column prop="cryptoAmount" :label="t('payment.cryptoAmount')" width="120">
              <template #default="{ row }">
                {{ formatAmount(row.cryptoAmount, 6) }}
              </template>
            </el-table-column>
            <el-table-column prop="paymentMethod" :label="t('payment.paymentMethod')" width="120" />
            <el-table-column prop="createdAt" :label="t('payment.createdAt')" width="160">
              <template #default="{ row }">
                {{ formatDateTime(row.createdAt) }}
              </template>
            </el-table-column>
          </el-table>

          <div class="pagination">
            <el-pagination
              :current-page="orderQuery.pageNum"
              :page-size="orderQuery.pageSize"
              :page-size-options="PAGE_SIZE_OPTIONS"
              :total="orderTotal"
              layout="total, sizes, prev, pager, next"
              @size-change="(size: number) => { orderQuery.pageSize = size; getOrderList() }"
              @current-change="handleOrderPageChange"
            />
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-dialog v-model="reviewDialogVisible" :title="t('payment.reviewKYC')" width="500px">
      <el-form :model="reviewForm" label-width="100px">
        <el-form-item :label="t('payment.reviewStatus')">
          <el-select v-model="reviewForm.status" style="width: 100%">
            <el-option label="Approve" value="APPROVED" />
            <el-option label="Reject" value="REJECTED" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('payment.reviewerId')">
          <el-input v-model="reviewForm.reviewerId" :placeholder="t('payment.reviewerId')" />
        </el-form-item>
        <el-form-item :label="t('payment.reviewNotes')">
          <el-input v-model="reviewForm.reviewNotes" type="textarea" :rows="3" :placeholder="t('payment.reviewNotes')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="reviewDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleReviewSubmit">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.payment-container {
  padding: 20px;
}

.statistics-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-item {
  text-align: center;
  padding: 10px;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.stat-value.warning {
  color: #e6a23c;
}

.stat-value.success {
  color: #67c23a;
}

.table-card {
  margin-top: 20px;
}

.toolbar {
  margin-bottom: 20px;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.positive {
  color: #67c23a;
}

.negative {
  color: #f56c6c;
}
</style>