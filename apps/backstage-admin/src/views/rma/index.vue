<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, View, Check, Edit, Close } from '@element-plus/icons-vue'
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
  type RmaQueryParam
} from '@/apis/rma'
import { formatDateTime } from '@/utils/datetime'
import { RMA_STATUS, STATUS_TAG_TYPE_MAP } from './constants'
import RmaFormDialog from './RmaFormDialog.vue'
import RmaDetailDialog from './RmaDetailDialog.vue'
import RmaInspectionDialog from './RmaInspectionDialog.vue'
import RmaDispositionDialog from './RmaDispositionDialog.vue'

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

const detailVisible = ref(false)
const detailData = ref<RmaOrder | null>(null)

const inspectionVisible = ref(false)

const dispositionVisible = ref(false)

const currentAction = ref('')
const currentRmaId = ref<number | null>(null)

const statusOptions = [
  { label: t('rma.statusPENDING'), value: RMA_STATUS.PENDING },
  { label: t('rma.statusAWAITING_INSPECTION'), value: RMA_STATUS.AWAITING_INSPECTION },
  { label: t('rma.statusINSPECTED'), value: RMA_STATUS.INSPECTED },
  { label: t('rma.statusAWAITING_DISPOSITION'), value: RMA_STATUS.AWAITING_DISPOSITION },
  { label: t('rma.statusDISPOSED'), value: RMA_STATUS.DISPOSED },
  { label: t('rma.statusCOMPLETED'), value: RMA_STATUS.COMPLETED },
  { label: t('rma.statusCANCELLED'), value: RMA_STATUS.CANCELLED }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getRmaListAPI(listQuery.value)
    list.value = response.data || []
    total.value = response.total || 0
  } catch (error: any) {
    console.error('Failed to load RMA orders:', error)
    ElMessage.error(error.message || 'Failed to load RMA orders')
  } finally {
    listLoading.value = false
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
  dialogVisible.value = true
}

const handleSubmit = async (form: {
  orderNo: string
  returnType: string
  customerId: number | undefined
  customerName: string
  contactPhone: string
  reasonCode: string
  reasonDescription: string
  warehouseId: number | undefined
  items: Array<{ skuCode: string; skuName: string; expectedQuantity: number; unitPrice: number }>
}) => {
  dialogLoading.value = true
  try {
    await createRmaAPI({
      orderNo: form.orderNo,
      returnType: form.returnType,
      customerId: form.customerId ?? '',
      customerName: form.customerName,
      contactPhone: form.contactPhone,
      reasonCode: form.reasonCode,
      reasonDescription: form.reasonDescription,
      warehouseId: form.warehouseId,
      items: form.items
    })
    ElMessage.success(t('rma.createSuccess'))
    dialogVisible.value = false
    getList()
  } catch (error: any) {
    console.error('Failed to create RMA order:', error)
    ElMessage.error(error.message || 'Failed to create RMA order')
  } finally {
    dialogLoading.value = false
  }
}

const handleView = async (row: RmaOrder) => {
  try {
    const response = await getRmaByIdAPI(row.id!)
    detailData.value = response.data
    detailVisible.value = true
  } catch (error: any) {
    console.error('Failed to load RMA detail:', error)
    ElMessage.error(error.message || 'Failed to load RMA detail')
  }
}

const handleSubmitInspection = async (row: RmaOrder) => {
  try {
    await ElMessageBox.confirm(t('rma.confirmSubmitInspection') || 'Submit this RMA for inspection?', t('common.confirm'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await submitRmaForInspectionAPI(row.id!)
    ElMessage.success(t('rma.submitSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to submit for inspection:', error)
      ElMessage.error(error.message || 'Failed to submit for inspection')
    }
  }
}

const handleRecordInspection = (row: RmaOrder) => {
  currentAction.value = 'inspection'
  currentRmaId.value = row.id ?? null
  inspectionVisible.value = true
}

const submitInspection = async (form: { inspector: string; result: string; conclusion: string; remark: string }) => {
  dialogLoading.value = true
  try {
    await recordRmaInspectionAPI(currentRmaId.value!, form)
    ElMessage.success(t('rma.recordSuccess'))
    inspectionVisible.value = false
    getList()
  } catch (error: any) {
    console.error('Failed to record inspection:', error)
    ElMessage.error(error.message || 'Failed to record inspection')
  } finally {
    dialogLoading.value = false
  }
}

const handleMakeDisposition = (row: RmaOrder) => {
  currentAction.value = 'disposition'
  currentRmaId.value = row.id ?? null
  dispositionVisible.value = true
}

const submitDisposition = async (form: { dispositionType: string; refundAmount: number; replacementSkuCode: string; replacementQuantity: number; remark: string }) => {
  dialogLoading.value = true
  try {
    await makeRmaDispositionAPI(currentRmaId.value!, form)
    ElMessage.success(t('rma.dispositionSuccess'))
    dispositionVisible.value = false
    getList()
  } catch (error: any) {
    console.error('Failed to make disposition:', error)
    ElMessage.error(error.message || 'Failed to make disposition')
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
    await executeRmaDispositionAPI(row.id!)
    ElMessage.success(t('rma.executeSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to execute disposition:', error)
      ElMessage.error(error.message || 'Failed to execute disposition')
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
    await cancelRmaAPI(row.id!)
    ElMessage.success(t('rma.cancelSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to cancel RMA:', error)
      ElMessage.error(error.message || 'Failed to cancel RMA')
    }
  }
}

const getStatusType = (status: string): string => {
  return STATUS_TAG_TYPE_MAP[status] || 'info'
}

const getStatusLabel = (status: string) => {
  return t(`rma.status${status}`)
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
              v-if="row.status === RMA_STATUS.PENDING"
              type="warning"
              size="small"
              :icon="Check"
              @click="handleSubmitInspection(row)"
            >
              {{ t('rma.submitInspection') }}
            </el-button>
            <el-button
              v-if="row.status === RMA_STATUS.AWAITING_INSPECTION"
              type="primary"
              size="small"
              :icon="Edit"
              @click="handleRecordInspection(row)"
            >
              {{ t('rma.recordInspection') }}
            </el-button>
            <el-button
              v-if="row.status === RMA_STATUS.INSPECTED"
              type="warning"
              size="small"
              :icon="Edit"
              @click="handleMakeDisposition(row)"
            >
              {{ t('rma.makeDisposition') }}
            </el-button>
            <el-button
              v-if="row.status === RMA_STATUS.AWAITING_DISPOSITION"
              type="success"
              size="small"
              :icon="Check"
              @click="handleExecuteDisposition(row)"
            >
              {{ t('rma.executeDisposition') }}
            </el-button>
            <el-button
              v-if="row.status === RMA_STATUS.PENDING || row.status === RMA_STATUS.AWAITING_INSPECTION"
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

    <RmaFormDialog
      v-model:visible="dialogVisible"
      :title="dialogTitle"
      :loading="dialogLoading"
      @submit="handleSubmit"
    />

    <RmaDetailDialog
      v-model:visible="detailVisible"
      :detail-data="detailData"
    />

    <RmaInspectionDialog
      v-model:visible="inspectionVisible"
      :loading="dialogLoading"
      @submit="submitInspection"
    />

    <RmaDispositionDialog
      v-model:visible="dispositionVisible"
      :loading="dialogLoading"
      @submit="submitDisposition"
    />
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
</style>
