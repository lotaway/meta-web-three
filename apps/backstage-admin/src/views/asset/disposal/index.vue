<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listDisposal,
  createDisposal,
  approveDisposal,
  rejectDisposal,
  listAssets,
  type FixedAssetDisposal,
  type AssetDisposalCreateParams,
  type ApproveDisposalParams,
  type FixedAsset
} from '@/apis/asset'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const loading = ref(false)
const dialogVisible = ref(false)
const approveDialogVisible = ref(false)
const rejectDialogVisible = ref(false)
const dialogTitle = ref('Add Disposal')
const isEdit = ref(false)
const currentId = ref<number | null>(null)

// List data
const disposalList = ref<FixedAssetDisposal[]>([])
const assetList = ref<FixedAsset[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)

// Search params
const searchParams = reactive({
  status: ''
})

// Form data
const formData = reactive<AssetDisposalCreateParams>({
  disposalCode: '',
  disposalType: '',
  assetId: 0,
  disposalAmount: 0,
  disposalDate: '',
  disposalReason: '',
  disposalMethod: '',
  acquirerName: '',
  acquirerContact: '',
  remark: ''
})

// Approve/Reject form
const approvalForm = reactive({
  approverId: 0,
  approverName: '',
  comment: ''
})

const formRef = ref()
const approvalFormRef = ref()

const statusOptions = [
  { value: 'PENDING_APPROVAL', label: 'Pending Approval' },
  { value: 'APPROVED', label: 'Approved' },
  { value: 'REJECTED', label: 'Rejected' },
  { value: 'COMPLETED', label: 'Completed' }
]

const disposalTypeOptions = [
  { value: 'SCRAP', label: 'Scrap' },
  { value: 'SALE', label: 'Sale' },
  { value: 'TRANSFER', label: 'Transfer' },
  { value: 'DAMAGE', label: 'Damage' }
]

const disposalMethodOptions = [
  { value: 'DISASSEMBLY', label: 'Disassembly' },
  { value: 'SCRAP', label: 'Scrap Sale' },
  { value: 'AUCTION', label: 'Auction' },
  { value: 'DONATION', label: 'Donation' },
  { value: 'TRANSFER', label: 'Transfer' }
]

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

const formatDate = (date: string) => {
  if (!date) return '-'
  return date.split('T')[0]
}

const getStatusLabel = (status: string) => {
  const map: Record<string, string> = {
    PENDING_APPROVAL: 'Pending Approval',
    APPROVED: 'Approved',
    REJECTED: 'Rejected',
    COMPLETED: 'Completed'
  }
  return map[status] || status
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING_APPROVAL: 'warning',
    APPROVED: 'success',
    REJECTED: 'danger',
    COMPLETED: 'info'
  }
  return map[status] || 'info'
}

const getDisposalTypeLabel = (type: string) => {
  const map: Record<string, string> = {
    SCRAP: 'Scrap',
    SALE: 'Sale',
    TRANSFER: 'Transfer',
    DAMAGE: 'Damage'
  }
  return map[type] || type
}

const loadAssets = async () => {
  try {
    const response = await listAssets({ status: 'IN_USE' })
    assetList.value = response.data || []
  } catch (error) {
    console.error('Failed to load assets:', error)
  }
}

const loadData = async () => {
  loading.value = true
  try {
    const params = {
      ...(searchParams.status && { status: searchParams.status })
    }
    const response = await listDisposal(params)
    disposalList.value = response.data || []
    total.value = (response.data || []).length
  } catch (error) {
    console.error('Failed to load disposal list:', error)
    ElMessage.error('Failed to load disposal list')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  currentPage.value = 1
  loadData()
}

const handleReset = () => {
  searchParams.status = ''
  currentPage.value = 1
  loadData()
}

const handlePageChange = (page: number) => {
  currentPage.value = page
  loadData()
}

const handleSizeChange = (size: number) => {
  pageSize.value = size
  currentPage.value = 1
  loadData()
}

const handleAdd = () => {
  dialogTitle.value = 'Add Disposal'
  isEdit.value = false
  currentId.value = null
  resetForm()
  dialogVisible.value = true
}

const handleAssetSelect = (assetId: number) => {
  const asset = assetList.value.find(a => a.id === assetId)
  if (asset) {
    formData.disposalAmount = asset.netValue || 0
  }
}

const handleApprove = (row: FixedAssetDisposal) => {
  currentId.value = row.id
  approvalForm.approverId = 0
  approvalForm.approverName = ''
  approvalForm.comment = ''
  approveDialogVisible.value = true
}

const handleReject = (row: FixedAssetDisposal) => {
  currentId.value = row.id
  approvalForm.approverId = 0
  approvalForm.approverName = ''
  approvalForm.comment = ''
  rejectDialogVisible.value = true
}

const submitForm = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid: boolean) => {
    if (!valid) return
    try {
      await createDisposal(formData)
      ElMessage.success('Disposal created successfully')
      dialogVisible.value = false
      loadData()
    } catch (error) {
      console.error('Failed to create disposal:', error)
      ElMessage.error('Failed to create disposal')
    }
  })
}

const submitApprove = async () => {
  if (!currentId.value) return
  try {
    await approveDisposal(currentId.value, approvalForm as ApproveDisposalParams)
    ElMessage.success('Disposal approved successfully')
    approveDialogVisible.value = false
    loadData()
  } catch (error) {
    console.error('Failed to approve disposal:', error)
    ElMessage.error('Failed to approve disposal')
  }
}

const submitReject = async () => {
  if (!currentId.value) return
  try {
    await rejectDisposal(currentId.value, approvalForm as ApproveDisposalParams)
    ElMessage.success('Disposal rejected')
    rejectDialogVisible.value = false
    loadData()
  } catch (error) {
    console.error('Failed to reject disposal:', error)
    ElMessage.error('Failed to reject disposal')
  }
}

const resetForm = () => {
  formData.disposalCode = ''
  formData.disposalType = ''
  formData.assetId = 0
  formData.disposalAmount = 0
  formData.disposalDate = ''
  formData.disposalReason = ''
  formData.disposalMethod = ''
  formData.acquirerName = ''
  formData.acquirerContact = ''
  formData.remark = ''
}

const cancelDialog = () => {
  dialogVisible.value = false
  resetForm()
}

onMounted(() => {
  loadData()
  loadAssets()
})
</script>

<template>
  <div class="disposal-container">
    <!-- Search Bar -->
    <el-card class="search-card">
      <el-form :model="searchParams" inline>
        <el-form-item label="Status">
          <el-select v-model="searchParams.status" placeholder="Select Status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Action Bar -->
    <el-card class="table-card">
      <div class="table-header">
        <el-button type="primary" @click="handleAdd">Add Disposal</el-button>
      </div>

      <!-- Table -->
      <el-table :data="disposalList" v-loading="loading" stripe>
        <el-table-column prop="disposalCode" label="Disposal Code" width="140" />
        <el-table-column prop="assetCode" label="Asset Code" width="120" />
        <el-table-column prop="assetName" label="Asset Name" min-width="150" />
        <el-table-column prop="disposalType" label="Type" width="100">
          <template #default="{ row }">
            {{ getDisposalTypeLabel(row.disposalType) }}
          </template>
        </el-table-column>
        <el-table-column prop="originalValue" label="Original Value" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.originalValue) }}
          </template>
        </el-table-column>
        <el-table-column prop="netValue" label="Net Value" width="120" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.netValue) }}
          </template>
        </el-table-column>
        <el-table-column prop="disposalAmount" label="Disposal Amount" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.disposalAmount) }}
          </template>
        </el-table-column>
        <el-table-column prop="gainLoss" label="Gain/Loss" width="100" align="right">
          <template #default="{ row }">
            <span :style="{ color: row.gainLoss >= 0 ? '#67C23A' : '#F56C6C' }">
              ¥{{ formatMoney(row.gainLoss) }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="disposalDate" label="Date" width="100">
          <template #default="{ row }">
            {{ formatDate(row.disposalDate) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="180" fixed="right">
          <template #default="{ row }">
            <el-button v-if="row.status === 'PENDING_APPROVAL'" type="success" link @click="handleApprove(row)">Approve</el-button>
            <el-button v-if="row.status === 'PENDING_APPROVAL'" type="danger" link @click="handleReject(row)">Reject</el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- Pagination -->
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @current-change="handlePageChange"
        @size-change="handleSizeChange"
        class="pagination"
      />
    </el-card>

    <!-- Add Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="700px" destroy-on-close>
      <el-form ref="formRef" :model="formData" label-width="140px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Disposal Code">
              <el-input v-model="formData.disposalCode" placeholder="Enter code" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Asset">
              <el-select v-model="formData.assetId" placeholder="Select asset" style="width: 100%" @change="handleAssetSelect">
                <el-option v-for="asset in assetList" :key="asset.id" :label="`${asset.assetCode} - ${asset.assetName}`" :value="asset.id" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Disposal Type">
              <el-select v-model="formData.disposalType" placeholder="Select type" style="width: 100%">
                <el-option v-for="item in disposalTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Disposal Date">
              <el-date-picker v-model="formData.disposalDate" type="date" placeholder="Select date" value-format="YYYY-MM-DD" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Disposal Amount">
              <el-input-number v-model="formData.disposalAmount" :min="0" :precision="2" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Disposal Method">
              <el-select v-model="formData.disposalMethod" placeholder="Select method" style="width: 100%">
                <el-option v-for="item in disposalMethodOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item label="Disposal Reason">
              <el-input v-model="formData.disposalReason" type="textarea" :rows="2" placeholder="Enter reason" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Acquirer Name">
              <el-input v-model="formData.acquirerName" placeholder="Enter name" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Acquirer Contact">
              <el-input v-model="formData.acquirerContact" placeholder="Enter contact" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item label="Remark">
              <el-input v-model="formData.remark" type="textarea" :rows="2" placeholder="Enter remark" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="cancelDialog">Cancel</el-button>
        <el-button type="primary" @click="submitForm">Confirm</el-button>
      </template>
    </el-dialog>

    <!-- Approve Dialog -->
    <el-dialog v-model="approveDialogVisible" title="Approve Disposal" width="500px" destroy-on-close>
      <el-form :model="approvalForm" label-width="140px">
        <el-form-item label="Approver ID">
          <el-input-number v-model="approvalForm.approverId" :min="0" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Approver Name">
          <el-input v-model="approvalForm.approverName" placeholder="Enter name" />
        </el-form-item>
        <el-form-item label="Comment">
          <el-input v-model="approvalForm.comment" type="textarea" :rows="3" placeholder="Enter comment" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="approveDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="submitApprove">Approve</el-button>
      </template>
    </el-dialog>

    <!-- Reject Dialog -->
    <el-dialog v-model="rejectDialogVisible" title="Reject Disposal" width="500px" destroy-on-close>
      <el-form :model="approvalForm" label-width="140px">
        <el-form-item label="Approver ID">
          <el-input-number v-model="approvalForm.approverId" :min="0" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Approver Name">
          <el-input v-model="approvalForm.approverName" placeholder="Enter name" />
        </el-form-item>
        <el-form-item label="Reason">
          <el-input v-model="approvalForm.comment" type="textarea" :rows="3" placeholder="Enter reject reason" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="rejectDialogVisible = false">Cancel</el-button>
        <el-button type="danger" @click="submitReject">Reject</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.disposal-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card {
  margin-bottom: 20px;
}

.table-header {
  margin-bottom: 15px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>