<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Check, Close } from '@element-plus/icons-vue'
import {
  getProcurementListAPI,
  createProcurementOrderAPI,
  approveProcurementOrderAPI,
  rejectProcurementOrderAPI,
  type ProcurementOrder,
  type ProcurementQueryParam
} from '@/apis/procurement'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<ProcurementQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<ProcurementOrder[]>([])
const listLoading = ref(true)
const total = ref(0)

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const isEdit = ref(false)

const formData = ref<ProcurementOrder>({
  orderNo: '',
  supplierCode: '',
  supplierName: '',
  warehouseId: undefined,
  warehouseName: '',
  purchaseType: '',
  status: 'PENDING',
  totalAmount: 0,
  currency: 'CNY',
  paymentTerms: '',
  deliveryTerms: '',
  remark: ''
})

const statusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'Approved', value: 'APPROVED' },
  { label: 'Rejected', value: 'REJECTED' },
  { label: 'Completed', value: 'COMPLETED' },
  { label: 'Cancelled', value: 'CANCELLED' }
]

const purchaseTypeOptions = [
  { label: 'Standard Purchase', value: 'STANDARD' },
  { label: 'Urgent Purchase', value: 'URGENT' },
  { label: 'Contract Purchase', value: 'CONTRACT' }
]

const currencyOptions = [
  { label: 'CNY', value: 'CNY' },
  { label: 'USD', value: 'USD' },
  { label: 'EUR', value: 'EUR' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getProcurementListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load procurement orders')
  }
}

onMounted(() => {
  getList()
})

const handleSearch = () => {
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleAdd = () => {
  isEdit.value = false
  formData.value = {
    orderNo: '',
    supplierCode: '',
    supplierName: '',
    warehouseId: undefined,
    warehouseName: '',
    purchaseType: 'STANDARD',
    status: 'PENDING',
    totalAmount: 0,
    currency: 'CNY',
    paymentTerms: '',
    deliveryTerms: '',
    remark: ''
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await createProcurementOrderAPI(formData.value)
    ElMessage.success('Procurement order created successfully')
    dialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to create procurement order')
  } finally {
    dialogLoading.value = false
  }
}

const handleApprove = async (row: ProcurementOrder) => {
  try {
    await ElMessageBox.confirm('Approve this procurement order?', 'Confirm', {
      confirmButtonText: 'Approve',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await approveProcurementOrderAPI(row.orderNo, 'Admin')
    ElMessage.success('Order approved successfully')
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to approve order')
    }
  }
}

const handleReject = async (row: ProcurementOrder) => {
  try {
    const { value: reason } = await ElMessageBox.prompt('Enter rejection reason:', 'Reject Order', {
      confirmButtonText: 'Reject',
      cancelButtonText: 'Cancel',
      inputPattern: /.+/,
      inputErrorMessage: 'Reason is required'
    })
    await rejectProcurementOrderAPI(row.orderNo, reason)
    ElMessage.success('Order rejected successfully')
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to reject order')
    }
  }
}

const getStatusLabel = (status: string) => {
  const option = statusOptions.find(s => s.value === status)
  return option?.label || status
}

type StatusTagType = 'primary' | 'success' | 'warning' | 'danger' | 'info'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    PENDING: 'warning',
    APPROVED: 'success',
    REJECTED: 'danger',
    COMPLETED: 'primary',
    CANCELLED: 'info'
  }
  return statusMap[status] || 'info'
}

const formatAmount = (amount?: number, currency?: string) => {
  if (!amount) return '-'
  return currency ? `${currency} ${amount.toFixed(2)}` : `$${amount.toFixed(2)}`
}
</script>

<template>
  <div class="procurement-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Order No">
          <el-input v-model="listQuery.orderNo" placeholder="Enter order number" clearable />
        </el-form-item>
        <el-form-item label="Supplier Code">
          <el-input v-model="listQuery.supplierCode" placeholder="Enter supplier code" clearable />
        </el-form-item>
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option
              v-for="item in statusOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
          <el-button type="success" :icon="Plus" @click="handleAdd">Create Order</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column label="Order No" prop="orderNo" min-width="140" />
        <el-table-column label="Supplier" min-width="150">
          <template #default="{ row }">
            <div>{{ row.supplierName }}</div>
            <div class="text-xs text-gray-500">{{ row.supplierCode }}</div>
          </template>
        </el-table-column>
        <el-table-column label="Warehouse" prop="warehouseName" min-width="120" />
        <el-table-column label="Purchase Type" prop="purchaseType" min-width="120">
          <template #default="{ row }">
            {{ row.purchaseType || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Status" min-width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Total Amount" min-width="120">
          <template #default="{ row }">
            {{ formatAmount(row.totalAmount, row.currency) }}
          </template>
        </el-table-column>
        <el-table-column label="Expected Delivery" min-width="160">
          <template #default="{ row }">
            {{ row.expectedDeliveryDate ? formatDateTime(row.expectedDeliveryDate) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Create Time" min-width="160">
          <template #default="{ row }">
            {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="180" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="row.status === 'PENDING'"
              type="success"
              size="small"
              :icon="Check"
              @click="handleApprove(row)"
            >
              Approve
            </el-button>
            <el-button
              v-if="row.status === 'PENDING'"
              type="danger"
              size="small"
              :icon="Close"
              @click="handleReject(row)"
            >
              Reject
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      title="Create Procurement Order"
      width="600px"
      :close-on-click-modal="false"
    >
      <el-form :model="formData" label-width="140px">
        <el-form-item label="Order No" required>
          <el-input v-model="formData.orderNo" placeholder="Auto-generated if empty" />
        </el-form-item>
        <el-form-item label="Supplier Code" required>
          <el-input v-model="formData.supplierCode" placeholder="Enter supplier code" />
        </el-form-item>
        <el-form-item label="Supplier Name" required>
          <el-input v-model="formData.supplierName" placeholder="Enter supplier name" />
        </el-form-item>
        <el-form-item label="Warehouse">
          <el-input v-model="formData.warehouseName" placeholder="Enter warehouse name" />
        </el-form-item>
        <el-form-item label="Purchase Type">
          <el-select v-model="formData.purchaseType" placeholder="Select type">
            <el-option
              v-for="item in purchaseTypeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Currency">
          <el-select v-model="formData.currency" placeholder="Select currency">
            <el-option
              v-for="item in currencyOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Total Amount">
          <el-input-number v-model="formData.totalAmount" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item label="Payment Terms">
          <el-input v-model="formData.paymentTerms" placeholder="e.g., Net 30" />
        </el-form-item>
        <el-form-item label="Delivery Terms">
          <el-input v-model="formData.deliveryTerms" placeholder="e.g., FOB Shanghai" />
        </el-form-item>
        <el-form-item label="Remark">
          <el-input v-model="formData.remark" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.procurement-container {
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

.text-xs {
  font-size: 12px;
}

.text-gray-500 {
  color: #999;
}
</style>