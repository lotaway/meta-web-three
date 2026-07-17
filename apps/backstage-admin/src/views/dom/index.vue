<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, View, Edit, Check, Close } from '@element-plus/icons-vue'
import {
  getDomListAPI,
  getDomByIdAPI,
  createDomOrderAPI,
  checkDomAvailabilityAPI,
  sourceDomOrderAPI,
  approveDomFulfillmentAPI,
  cancelDomOrderAPI,
  type DomOrder,
  type DomQueryParam
} from '@/apis/dom'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<DomQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<DomOrder[]>([])
const listLoading = ref(true)
const total = ref(0)

const dialogVisible = ref(false)
const dialogTitle = ref('')
const dialogLoading = ref(false)

const detailVisible = ref(false)
const detailData = ref<DomOrder | null>(null)

const formData = ref<DomOrder>({
  domOrderNo: '',
  originalOrderNo: '',
  customerId: undefined,
  customerName: '',
  region: '',
  sourcingStrategy: 'NEAREST_WAREHOUSE',
  status: 'PENDING',
  lines: []
})

const strategyOptions = [
  { label: t('dom.strategyNEAREST_WAREHOUSE'), value: 'NEAREST_WAREHOUSE' },
  { label: t('dom.strategyLOWEST_COST'), value: 'LOWEST_COST' },
  { label: t('dom.strategyBALANCED'), value: 'BALANCED' }
]

const statusOptions = [
  { label: t('dom.statusPENDING'), value: 'PENDING' },
  { label: t('dom.statusATP_FAILED'), value: 'ATP_FAILED' },
  { label: t('dom.statusSOURCING'), value: 'SOURCING' },
  { label: t('dom.statusSOURCING_COMPLETED'), value: 'SOURCING_COMPLETED' },
  { label: t('dom.statusFULFILLED'), value: 'FULFILLED' },
  { label: t('dom.statusCANCELLED'), value: 'CANCELLED' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getDomListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load DOM orders')
  }
}

onMounted(() => {
  getList()
})

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

const handlePageChange = (page: number) => {
  listQuery.value.pageNum = page
  getList()
}

const handleSizeChange = (size: number) => {
  listQuery.value.pageSize = size
  listQuery.value.pageNum = 1
  getList()
}

const handleAdd = () => {
  dialogTitle.value = t('dom.createDom')
  formData.value = {
    domOrderNo: '',
    originalOrderNo: '',
    customerId: undefined,
    customerName: '',
    region: '',
    sourcingStrategy: 'NEAREST_WAREHOUSE',
    status: 'PENDING',
    lines: []
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await createDomOrderAPI({
      originalOrderNo: formData.value.originalOrderNo,
      customerId: formData.value.customerId,
      customerName: formData.value.customerName,
      region: formData.value.region,
      sourcingStrategy: formData.value.sourcingStrategy,
      items: formData.value.lines
    })
    ElMessage.success(t('dom.createSuccess'))
    dialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to create DOM order')
  } finally {
    dialogLoading.value = false
  }
}

const handleView = async (row: DomOrder) => {
  try {
    const response = await getDomByIdAPI(row.id)
    detailData.value = response.data
    detailVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load DOM detail')
  }
}

const handleCheckAtp = async (row: DomOrder) => {
  try {
    await ElMessageBox.confirm(t('dom.confirmCheckAtp') || 'Check ATP for this order?', t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await checkDomAvailabilityAPI(row.id)
    ElMessage.success(t('dom.atpCheckSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to check ATP')
    }
  }
}

const handleSourceOrder = async (row: DomOrder) => {
  try {
    await ElMessageBox.confirm(t('dom.confirmSource') || 'Source this order?', t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await sourceDomOrderAPI(row.id)
    ElMessage.success(t('dom.sourceSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to source order')
    }
  }
}

const handleApproveFulfillment = async (row: DomOrder) => {
  try {
    await ElMessageBox.confirm(t('dom.confirmApprove') || 'Approve fulfillment plan?', t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await approveDomFulfillmentAPI(row.id)
    ElMessage.success(t('dom.approveSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to approve fulfillment')
    }
  }
}

const handleCancel = async (row: DomOrder) => {
  try {
    await ElMessageBox.confirm(t('dom.confirmCancel'), t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await cancelDomOrderAPI(row.id)
    ElMessage.success(t('dom.cancelSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to cancel DOM order')
    }
  }
}

const addItem = () => {
  if (!formData.value.lines) {
    formData.value.lines = []
  }
  formData.value.lines.push({
    skuCode: '',
    skuName: '',
    quantity: 1,
    unitPrice: 0
  })
}

const removeItem = (index: number) => {
  formData.value.lines?.splice(index, 1)
}

type StatusTagType = 'primary' | 'success' | 'warning' | 'danger' | 'info'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    PENDING: 'info',
    ATP_FAILED: 'danger',
    SOURCING: 'warning',
    SOURCING_COMPLETED: 'primary',
    FULFILLED: 'success',
    CANCELLED: 'info'
  }
  return statusMap[status] || 'info'
}

const getStatusLabel = (status: string) => {
  const key = `dom.status${status}`
  return t(key)
}

const formatAmount = (amount?: number) => {
  if (!amount) return '-'
  return `$${amount.toFixed(2)}`
}
</script>

<template>
  <div class="dom-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item :label="t('dom.domOrderNo')">
          <el-input v-model="listQuery.domOrderNo" :placeholder="t('common.placeholderSuffix') + t('dom.domOrderNo')" clearable />
        </el-form-item>
        <el-form-item :label="t('dom.status')">
          <el-select v-model="listQuery.status" :placeholder="t('common.selectPlaceholder')" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
          <el-button type="success" :icon="Plus" @click="handleAdd">{{ t('dom.createDom') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column :label="t('dom.domOrderNo')" prop="domOrderNo" min-width="140" />
        <el-table-column :label="t('dom.originalOrderNo')" prop="originalOrderNo" min-width="140" />
        <el-table-column :label="t('dom.customerName')" prop="customerName" min-width="120" />
        <el-table-column :label="t('dom.status')" min-width="130">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('dom.sourcingStrategy')" min-width="130">
          <template #default="{ row }">
            {{ t(`dom.strategy${row.sourcingStrategy}`) || row.sourcingStrategy || '-' }}
          </template>
        </el-table-column>
        <el-table-column :label="t('dom.region')" prop="region" min-width="100" />
        <el-table-column :label="t('dom.totalAmount')" min-width="100">
          <template #default="{ row }">
            {{ formatAmount(row.totalAmount) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('common.createdAt') || 'Created At'" min-width="160">
          <template #default="{ row }">
            {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
          </template>
        </el-table-column>
        <el-table-column :label="t('common.operations')" width="300" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" :icon="View" @click="handleView(row)">
              {{ t('common.detail') }}
            </el-button>
            <el-button
              v-if="row.status === 'PENDING'"
              type="warning"
              size="small"
              :icon="Check"
              @click="handleCheckAtp(row)"
            >
              {{ t('dom.checkAtp') }}
            </el-button>
            <el-button
              v-if="row.status === 'SOURCING'"
              type="primary"
              size="small"
              :icon="Edit"
              @click="handleSourceOrder(row)"
            >
              {{ t('dom.sourceOrder') }}
            </el-button>
            <el-button
              v-if="row.status === 'SOURCING_COMPLETED'"
              type="success"
              size="small"
              :icon="Check"
              @click="handleApproveFulfillment(row)"
            >
              {{ t('dom.approveFulfillment') }}
            </el-button>
            <el-button
              v-if="row.status === 'PENDING' || row.status === 'ATP_FAILED'"
              type="danger"
              size="small"
              :icon="Close"
              @click="handleCancel(row)"
            >
              {{ t('dom.cancelDom') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        v-model:current-page="listQuery.pageNum"
        v-model:page-size="listQuery.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @current-change="handlePageChange"
        @size-change="handleSizeChange"
      />
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="700px"
      :close-on-click-modal="false"
    >
      <el-form :model="formData" label-width="140px">
        <el-form-item :label="t('dom.originalOrderNo')" required>
          <el-input v-model="formData.originalOrderNo" :placeholder="t('common.placeholderSuffix') + t('dom.originalOrderNo')" />
        </el-form-item>
        <el-form-item :label="t('dom.customerName')" required>
          <el-input v-model="formData.customerName" :placeholder="t('common.placeholderSuffix') + t('dom.customerName')" />
        </el-form-item>
        <el-form-item :label="t('dom.region')">
          <el-input v-model="formData.region" :placeholder="t('common.placeholderSuffix') + t('dom.region')" />
        </el-form-item>
        <el-form-item :label="t('dom.sourcingStrategy')" required>
          <el-select v-model="formData.sourcingStrategy">
            <el-option v-for="item in strategyOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Items" required>
          <div class="items-wrapper">
            <div v-for="(item, index) in formData.lines" :key="index" class="item-row">
              <el-input v-model="item.skuCode" placeholder="SKU Code" style="width: 120px" />
              <el-input v-model="item.skuName" placeholder="SKU Name" style="width: 140px" />
              <el-input-number v-model="item.quantity" :min="1" :max="99999" size="small" />
              <el-input-number v-model="item.unitPrice" :min="0" :precision="2" size="small" />
              <el-button type="danger" size="small" :icon="Close" @click="removeItem(index)" />
            </div>
            <el-button type="primary" size="small" @click="addItem">+ Add Item</el-button>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="detailVisible"
      :title="t('dom.viewDom')"
      width="800px"
      :close-on-click-modal="false"
    >
      <template v-if="detailData">
        <el-descriptions :column="2" border>
          <el-descriptions-item :label="t('dom.domOrderNo')">{{ detailData.domOrderNo }}</el-descriptions-item>
          <el-descriptions-item :label="t('dom.originalOrderNo')">{{ detailData.originalOrderNo }}</el-descriptions-item>
          <el-descriptions-item :label="t('dom.customerName')">{{ detailData.customerName }}</el-descriptions-item>
          <el-descriptions-item :label="t('dom.region')">{{ detailData.region || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('dom.sourcingStrategy')">{{ t(`dom.strategy${detailData.sourcingStrategy}`) || detailData.sourcingStrategy }}</el-descriptions-item>
          <el-descriptions-item :label="t('dom.status')">
            <el-tag :type="getStatusType(detailData.status)">{{ getStatusLabel(detailData.status) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('dom.totalAmount')">{{ formatAmount(detailData.totalAmount) }}</el-descriptions-item>
        </el-descriptions>
        <el-table :data="detailData.lines" border stripe class="detail-items-table">
          <el-table-column label="SKU Code" prop="skuCode" min-width="120" />
          <el-table-column label="SKU Name" prop="skuName" min-width="140" />
          <el-table-column label="Quantity" prop="quantity" min-width="80" />
          <el-table-column label="Unit Price" min-width="100">
            <template #default="{ row }">{{ formatAmount(row.unitPrice) }}</template>
          </el-table-column>
        </el-table>
        <div v-if="detailData.fulfillmentPlan && detailData.fulfillmentPlan.length > 0" class="fulfillment-section">
          <h4>Fulfillment Plans</h4>
          <el-table :data="detailData.fulfillmentPlan" border stripe>
            <el-table-column label="Warehouse" prop="warehouseName" min-width="120" />
            <el-table-column label="SKU Code" prop="skuCode" min-width="120" />
            <el-table-column label="Quantity" prop="quantity" min-width="80" />
            <el-table-column label="Status" prop="status" min-width="100" />
          </el-table>
        </div>
      </template>
      <template #footer>
        <el-button @click="detailVisible = false">{{ t('common.close') || 'Close' }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.dom-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.search-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.table-card {
  min-height: 400px;
}

.detail-items-table {
  margin-top: 16px;
}

.fulfillment-section {
  margin-top: 16px;
}

.fulfillment-section h4 {
  margin-bottom: 8px;
}

.items-wrapper {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.item-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
</style>
