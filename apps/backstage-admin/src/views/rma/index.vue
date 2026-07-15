<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, View, Check, Close, Edit, Delete } from '@element-plus/icons-vue'
import {
  getRmaListAPI,
  getRmaByIdAPI,
  createRmaAPI,
  submitRmaForInspectionAPI,
  recordRmaInspectionAPI,
  makeRmaDispositionAPI,
  executeRmaDispositionAPI,
  cancelRmaAPI,
  type RmaOrder,
  type RmaQueryParam,
  type RmaInspection,
  type RmaDisposition
} from '@/apis/rma'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<RmaQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<RmaOrder[]>([])
const listLoading = ref(true)
const total = ref(0)

const dialogVisible = ref(false)
const dialogTitle = ref('')
const dialogLoading = ref(false)
const currentAction = ref('')

const detailVisible = ref(false)
const detailData = ref<RmaOrder | null>(null)

const inspectionVisible = ref(false)
const inspectionForm = ref<RmaInspection>({
  inspector: '',
  result: 'PASS',
  conclusion: '',
  remark: ''
})

const dispositionVisible = ref(false)
const dispositionForm = ref<RmaDisposition>({
  dispositionType: 'REFUND',
  refundAmount: 0,
  replacementSkuCode: '',
  replacementQuantity: 0,
  remark: ''
})

const formData = ref<RmaOrder>({
  rmaNo: '',
  orderNo: '',
  returnType: 'REFUND',
  customerId: undefined,
  customerName: '',
  contactPhone: '',
  reasonCode: '',
  reasonDescription: '',
  warehouseId: undefined,
  status: 'PENDING',
  items: []
})

const returnTypeOptions = [
  { label: t('rma.returnTypeREFUND'), value: 'REFUND' },
  { label: t('rma.returnTypeREPLACEMENT'), value: 'REPLACEMENT' },
  { label: t('rma.returnTypeREPAIR'), value: 'REPAIR' }
]

const statusOptions = [
  { label: t('rma.statusPENDING'), value: 'PENDING' },
  { label: t('rma.statusAWAITING_INSPECTION'), value: 'AWAITING_INSPECTION' },
  { label: t('rma.statusINSPECTED'), value: 'INSPECTED' },
  { label: t('rma.statusAWAITING_DISPOSITION'), value: 'AWAITING_DISPOSITION' },
  { label: t('rma.statusDISPOSED'), value: 'DISPOSED' },
  { label: t('rma.statusCOMPLETED'), value: 'COMPLETED' },
  { label: t('rma.statusCANCELLED'), value: 'CANCELLED' }
]

const inspectionResultOptions = [
  { label: 'PASS', value: 'PASS' },
  { label: 'FAIL', value: 'FAIL' },
  { label: 'PARTIAL', value: 'PARTIAL' }
]

const dispositionTypeOptions = [
  { label: t('rma.returnTypeREFUND'), value: 'REFUND' },
  { label: t('rma.returnTypeREPLACEMENT'), value: 'REPLACEMENT' },
  { label: t('rma.returnTypeREPAIR'), value: 'REPAIR' },
  { label: 'SCRAP', value: 'SCRAP' },
  { label: 'RETURN_TO_SUPPLIER', value: 'RETURN_TO_SUPPLIER' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getRmaListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.total || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load RMA orders')
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
  currentAction.value = 'create'
  dialogTitle.value = t('rma.createRma')
  formData.value = {
    rmaNo: '',
    orderNo: '',
    returnType: 'REFUND',
    customerId: undefined,
    customerName: '',
    contactPhone: '',
    reasonCode: '',
    reasonDescription: '',
    warehouseId: undefined,
    status: 'PENDING',
    items: []
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await createRmaAPI(formData.value)
    ElMessage.success(t('rma.createSuccess'))
    dialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to create RMA order')
  } finally {
    dialogLoading.value = false
  }
}

const handleView = async (row: RmaOrder) => {
  try {
    const response = await getRmaByIdAPI(row.rmaNo)
    detailData.value = response.data
    detailVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load RMA detail')
  }
}

const handleSubmitInspection = async (row: RmaOrder) => {
  try {
    await ElMessageBox.confirm(t('rma.confirmSubmitInspection') || 'Submit this RMA for inspection?', t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await submitRmaForInspectionAPI(row.rmaNo)
    ElMessage.success(t('rma.submitSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to submit for inspection')
    }
  }
}

const handleRecordInspection = async (row: RmaOrder) => {
  currentAction.value = 'inspection'
  dialogTitle.value = t('rma.recordInspection')
  inspectionForm.value = {
    inspector: '',
    result: 'PASS',
    conclusion: '',
    remark: ''
  }
  inspectionVisible.value = true
}

const submitInspection = async () => {
  dialogLoading.value = true
  try {
    await recordRmaInspectionAPI(currentRmaNo.value, inspectionForm.value)
    ElMessage.success(t('rma.recordSuccess'))
    inspectionVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to record inspection')
  } finally {
    dialogLoading.value = false
  }
}

const currentRmaNo = ref('')

const handleMakeDisposition = (row: RmaOrder) => {
  currentAction.value = 'disposition'
  currentRmaNo.value = row.rmaNo
  dialogTitle.value = t('rma.makeDisposition')
  dispositionForm.value = {
    dispositionType: 'REFUND',
    refundAmount: 0,
    replacementSkuCode: '',
    replacementQuantity: 0,
    remark: ''
  }
  dispositionVisible.value = true
}

const submitDisposition = async () => {
  dialogLoading.value = true
  try {
    await makeRmaDispositionAPI(currentRmaNo.value, dispositionForm.value)
    ElMessage.success(t('rma.dispositionSuccess'))
    dispositionVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to make disposition')
  } finally {
    dialogLoading.value = false
  }
}

const handleExecuteDisposition = async (row: RmaOrder) => {
  try {
    await ElMessageBox.confirm(t('rma.confirmExecuteDisposition') || 'Execute this disposition?', t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await executeRmaDispositionAPI(row.rmaNo)
    ElMessage.success(t('rma.executeSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to execute disposition')
    }
  }
}

const handleCancel = async (row: RmaOrder) => {
  try {
    await ElMessageBox.confirm(t('rma.confirmCancel'), t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await cancelRmaAPI(row.rmaNo)
    ElMessage.success(t('rma.cancelSuccess'))
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to cancel RMA')
    }
  }
}

const addItem = () => {
  if (!formData.value.items) {
    formData.value.items = []
  }
  formData.value.items.push({
    skuCode: '',
    skuName: '',
    expectedQuantity: 1,
    unitPrice: 0
  })
}

const removeItem = (index: number) => {
  formData.value.items?.splice(index, 1)
}

type StatusTagType = 'primary' | 'success' | 'warning' | 'danger' | 'info'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    PENDING: 'info',
    AWAITING_INSPECTION: 'warning',
    INSPECTED: 'primary',
    AWAITING_DISPOSITION: 'warning',
    DISPOSED: 'success',
    COMPLETED: 'success',
    CANCELLED: 'danger'
  }
  return statusMap[status] || 'info'
}

const getStatusLabel = (status: string) => {
  const key = `rma.status${status}`
  return t(key)
}

const formatAmount = (amount?: number) => {
  if (!amount) return '-'
  return `$${amount.toFixed(2)}`
}
</script>

<template>
  <div class="rma-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item :label="t('rma.rmaNo')">
          <el-input v-model="listQuery.rmaNo" :placeholder="t('common.placeholderSuffix') + t('rma.rmaNo')" clearable />
        </el-form-item>
        <el-form-item :label="t('rma.orderNo')">
          <el-input v-model="listQuery.orderNo" :placeholder="t('common.placeholderSuffix') + t('rma.orderNo')" clearable />
        </el-form-item>
        <el-form-item :label="t('rma.status')">
          <el-select v-model="listQuery.status" :placeholder="t('common.selectPlaceholder')" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
          <el-button type="success" :icon="Plus" @click="handleAdd">{{ t('rma.createRma') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column :label="t('rma.rmaNo')" prop="rmaNo" min-width="140" />
        <el-table-column :label="t('rma.orderNo')" prop="orderNo" min-width="140" />
        <el-table-column :label="t('rma.returnType')" min-width="100">
          <template #default="{ row }">
            {{ t(`rma.returnType${row.returnType}`) || row.returnType || '-' }}
          </template>
        </el-table-column>
        <el-table-column :label="t('rma.customerName')" prop="customerName" min-width="120" />
        <el-table-column :label="t('rma.status')" min-width="130">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('rma.totalQuantity')" min-width="90">
          <template #default="{ row }">
            {{ row.totalQuantity ?? '-' }}
          </template>
        </el-table-column>
        <el-table-column :label="t('rma.createdAt') || 'Created At'" min-width="160">
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
              @click="handleSubmitInspection(row)"
            >
              {{ t('rma.submitInspection') }}
            </el-button>
            <el-button
              v-if="row.status === 'AWAITING_INSPECTION'"
              type="primary"
              size="small"
              :icon="Edit"
              @click="handleRecordInspection(row)"
            >
              {{ t('rma.recordInspection') }}
            </el-button>
            <el-button
              v-if="row.status === 'INSPECTED'"
              type="warning"
              size="small"
              :icon="Edit"
              @click="handleMakeDisposition(row)"
            >
              {{ t('rma.makeDisposition') }}
            </el-button>
            <el-button
              v-if="row.status === 'AWAITING_DISPOSITION'"
              type="success"
              size="small"
              :icon="Check"
              @click="handleExecuteDisposition(row)"
            >
              {{ t('rma.executeDisposition') }}
            </el-button>
            <el-button
              v-if="row.status === 'PENDING' || row.status === 'AWAITING_INSPECTION'"
              type="danger"
              size="small"
              :icon="Close"
              @click="handleCancel(row)"
            >
              {{ t('rma.cancelRma') }}
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
        <el-form-item :label="t('rma.orderNo')" required>
          <el-input v-model="formData.orderNo" :placeholder="t('common.placeholderSuffix') + t('rma.orderNo')" />
        </el-form-item>
        <el-form-item :label="t('rma.returnType')" required>
          <el-select v-model="formData.returnType" :placeholder="t('common.selectPlaceholder')">
            <el-option v-for="item in returnTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('rma.customerName')" required>
          <el-input v-model="formData.customerName" :placeholder="t('common.placeholderSuffix') + t('rma.customerName')" />
        </el-form-item>
        <el-form-item :label="t('rma.contactPhone')">
          <el-input v-model="formData.contactPhone" :placeholder="t('common.placeholderSuffix') + t('rma.contactPhone')" />
        </el-form-item>
        <el-form-item :label="t('rma.reasonCode')">
          <el-input v-model="formData.reasonCode" :placeholder="t('common.placeholderSuffix') + t('rma.reasonCode')" />
        </el-form-item>
        <el-form-item :label="t('rma.reasonDescription')">
          <el-input v-model="formData.reasonDescription" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="Items" required>
          <div class="items-wrapper">
            <div v-for="(item, index) in formData.items" :key="index" class="item-row">
              <el-input v-model="item.skuCode" placeholder="SKU Code" style="width: 120px" />
              <el-input v-model="item.skuName" placeholder="SKU Name" style="width: 140px" />
              <el-input-number v-model="item.expectedQuantity" :min="1" :max="99999" size="small" />
              <el-input-number v-model="item.unitPrice" :min="0" :precision="2" size="small" />
              <el-button type="danger" size="small" :icon="Delete" @click="removeItem(index)" />
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
      :title="t('rma.viewRma')"
      width="800px"
      :close-on-click-modal="false"
    >
      <template v-if="detailData">
        <el-descriptions :column="2" border>
          <el-descriptions-item :label="t('rma.rmaNo')">{{ detailData.rmaNo }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.orderNo')">{{ detailData.orderNo }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.returnType')">{{ t(`rma.returnType${detailData.returnType}`) || detailData.returnType }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.customerName')">{{ detailData.customerName }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.contactPhone')">{{ detailData.contactPhone || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.status')">
            <el-tag :type="getStatusType(detailData.status)">{{ getStatusLabel(detailData.status) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('rma.reasonCode')">{{ detailData.reasonCode || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.reasonDescription')">{{ detailData.reasonDescription || '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.totalQuantity')">{{ detailData.totalQuantity ?? '-' }}</el-descriptions-item>
          <el-descriptions-item :label="t('rma.totalAmount')">{{ formatAmount(detailData.totalAmount) }}</el-descriptions-item>
        </el-descriptions>
        <el-table :data="detailData.items" border stripe class="detail-items-table">
          <el-table-column label="SKU Code" prop="skuCode" min-width="120" />
          <el-table-column label="SKU Name" prop="skuName" min-width="140" />
          <el-table-column label="Expected Qty" prop="expectedQuantity" min-width="100" />
          <el-table-column label="Unit Price" min-width="100">
            <template #default="{ row }">{{ formatAmount(row.unitPrice) }}</template>
          </el-table-column>
        </el-table>
      </template>
      <template #footer>
        <el-button @click="detailVisible = false">{{ t('common.close') || 'Close' }}</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="inspectionVisible"
      :title="t('rma.recordInspection')"
      width="500px"
      :close-on-click-modal="false"
    >
      <el-form :model="inspectionForm" label-width="120px">
        <el-form-item :label="t('rma.inspector') || 'Inspector'" required>
          <el-input v-model="inspectionForm.inspector" placeholder="Enter inspector" />
        </el-form-item>
        <el-form-item label="Result" required>
          <el-select v-model="inspectionForm.result">
            <el-option v-for="item in inspectionResultOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Conclusion">
          <el-input v-model="inspectionForm.conclusion" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="Remark">
          <el-input v-model="inspectionForm.remark" type="textarea" :rows="2" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="inspectionVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="submitInspection">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="dispositionVisible"
      :title="t('rma.makeDisposition')"
      width="500px"
      :close-on-click-modal="false"
    >
      <el-form :model="dispositionForm" label-width="160px">
        <el-form-item :label="t('rma.dispositionType') || 'Disposition Type'" required>
          <el-select v-model="dispositionForm.dispositionType">
            <el-option v-for="item in dispositionTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('rma.refundAmount') || 'Refund Amount'">
          <el-input-number v-model="dispositionForm.refundAmount" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item :label="t('rma.replacementSkuCode') || 'Replacement SKU'">
          <el-input v-model="dispositionForm.replacementSkuCode" placeholder="Enter SKU code" />
        </el-form-item>
        <el-form-item :label="t('rma.replacementQuantity') || 'Replacement Qty'">
          <el-input-number v-model="dispositionForm.replacementQuantity" :min="0" />
        </el-form-item>
        <el-form-item label="Remark">
          <el-input v-model="dispositionForm.remark" type="textarea" :rows="2" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dispositionVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="submitDisposition">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.rma-container {
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
