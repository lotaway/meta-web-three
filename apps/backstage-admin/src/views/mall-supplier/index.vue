<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, View, Check, Close } from '@element-plus/icons-vue'
import {
  getMallSupplierListAPI,
  getMallSupplierByIdAPI,
  registerMallSupplierAPI,
  submitMallSupplierVerificationAPI,
  verifyMallSupplierAPI,
  getMallSupplierPerformanceAPI,
  updateMallSupplierScoreAPI,
  type MallSupplier,
  type MallSupplierRegistration,
} from '@/apis/mallSupplier'
import { formatDateTime } from '@/utils/datetime'

const list = ref<MallSupplier[]>([])
const listLoading = ref(false)

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const isView = ref(false)

const formData = ref<MallSupplierRegistration>({
  supplierName: '',
  contactPerson: '',
  contactPhone: '',
  contactEmail: '',
  address: '',
  businessLicense: '',
  legalPerson: '',
})

const statusOptions = [
  { label: 'Pending', value: 0 },
  { label: 'Active', value: 1 },
  { label: 'Suspended', value: 2 },
  { label: 'Disabled', value: 3 },
]

const verificationStatusOptions = [
  { label: 'Not Submitted', value: 0 },
  { label: 'Pending', value: 1 },
  { label: 'Approved', value: 2 },
  { label: 'Rejected', value: 3 },
]

const getList = async () => {
  listLoading.value = true
  try {
    const res = await getMallSupplierListAPI()
    list.value = (res as any)?.data || (res as any) || []
  } catch (error) {
    ElMessage.error('Failed to load suppliers')
  } finally {
    listLoading.value = false
  }
}

onMounted(() => {
  getList()
})

const handleAdd = () => {
  isView.value = false
  formData.value = {
    supplierName: '',
    contactPerson: '',
    contactPhone: '',
    contactEmail: '',
    address: '',
    businessLicense: '',
    legalPerson: '',
  }
  dialogVisible.value = true
}

const handleView = async (row: MallSupplier) => {
  if (!row.id) return
  try {
    const res = await getMallSupplierByIdAPI(row.id)
    formData.value = {
      supplierName: res.data.supplierName || '',
      contactPerson: res.data.contactPerson || '',
      contactPhone: res.data.contactPhone || '',
      contactEmail: res.data.contactEmail || '',
      address: res.data.address || '',
      businessLicense: res.data.businessLicense || '',
      legalPerson: res.data.legalPerson || '',
    }
    isView.value = true
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load supplier details')
  }
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await registerMallSupplierAPI(formData.value)
    ElMessage.success('Supplier registered successfully')
    dialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to register supplier')
  } finally {
    dialogLoading.value = false
  }
}

const handleSubmitVerification = async (row: MallSupplier) => {
  if (!row.id) return
  try {
    await ElMessageBox.confirm(
      `Submit ${row.supplierName} for verification?`,
      'Confirm',
      { confirmButtonText: 'Confirm', cancelButtonText: 'Cancel', type: 'warning' }
    )
    await submitMallSupplierVerificationAPI(row.id)
    ElMessage.success('Verification submitted')
    getList()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error('Failed to submit verification')
  }
}

const handleVerify = async (row: MallSupplier) => {
  if (!row.id) return
  try {
    await ElMessageBox.prompt(
      `Approve or reject ${row.supplierName}? Enter "approve" or "reject"`,
      'Verify Supplier',
      { confirmButtonText: 'Submit', cancelButtonText: 'Cancel', inputPlaceholder: 'approve / reject' }
    ).then(async ({ value }) => {
      if (!value || (value !== 'approve' && value !== 'reject')) {
        ElMessage.warning('Please enter "approve" or "reject"')
        return
      }
      await verifyMallSupplierAPI({
        supplierId: row.id!,
        approved: value === 'approve',
        reason: value === 'approve' ? 'Approved by admin' : 'Rejected by admin',
      })
      ElMessage.success('Verification completed')
      getList()
    })
  } catch (error) {
    if (error !== 'cancel') ElMessage.error('Failed to verify supplier')
  }
}

const handleScore = async (row: MallSupplier) => {
  if (!row.id) return
  try {
    await ElMessageBox.prompt(
      `Update score for ${row.supplierName} (current: ${row.score || 0})`,
      'Update Score',
      { confirmButtonText: 'Submit', cancelButtonText: 'Cancel', inputPlaceholder: 'Score delta (e.g. 5 or -3)' }
    ).then(async ({ value }) => {
      const delta = parseInt(value, 10)
      if (isNaN(delta)) {
        ElMessage.warning('Please enter a valid number')
        return
      }
      await updateMallSupplierScoreAPI(row.id!, delta)
      ElMessage.success('Score updated')
      getList()
    })
  } catch (error) {
    if (error !== 'cancel') ElMessage.error('Failed to update score')
  }
}

const handleViewPerformance = async (row: MallSupplier) => {
  if (!row.id) return
  try {
    const res = await getMallSupplierPerformanceAPI(row.id)
    const perf = res.data
    ElMessageBox.alert(
      `Orders: ${perf.orderCount}\n` +
      `On-Time Delivery: ${(perf.onTimeDeliveryRate * 100).toFixed(1)}%\n` +
      `Quality Score: ${perf.qualityScore}\n` +
      `Response Time: ${perf.responseTime}h\n` +
      `Overall Score: ${perf.overallScore}`,
      `Performance - ${row.supplierName}`,
      { confirmButtonText: 'OK' }
    )
  } catch (error) {
    ElMessage.error('Failed to load performance')
  }
}

const getStatusTag = (status: number): 'info' | 'success' | 'warning' | 'danger' => {
  const map: Record<number, 'info' | 'success' | 'warning' | 'danger'> = {
    0: 'info', 1: 'success', 2: 'warning', 3: 'danger',
  }
  return map[status] || 'info'
}

const getStatusText = (status: number): string => {
  const map: Record<number, string> = {
    0: 'Pending', 1: 'Active', 2: 'Suspended', 3: 'Disabled',
  }
  return map[status] || 'Unknown'
}

const getVerificationTag = (status: number): 'info' | 'warning' | 'success' | 'danger' => {
  const map: Record<number, 'info' | 'warning' | 'success' | 'danger'> = {
    0: 'info', 1: 'warning', 2: 'success', 3: 'danger',
  }
  return map[status] || 'info'
}

const getVerificationText = (status: number): string => {
  const map: Record<number, string> = {
    0: 'Not Submitted', 1: 'Pending', 2: 'Approved', 3: 'Rejected',
  }
  return map[status] || 'Unknown'
}

const getLevelText = (level: number | undefined): string => {
  const map: Record<number, string> = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D',
  }
  return level ? map[level] || String(level) : '-'
}
</script>

<template>
  <div class="mall-supplier-container">
    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="primary" :icon="Plus" @click="handleAdd">Register Supplier</el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="supplierCode" label="Code" width="130" />
        <el-table-column prop="supplierName" label="Name" min-width="150" />
        <el-table-column prop="contactPerson" label="Contact" width="110" />
        <el-table-column prop="contactPhone" label="Phone" width="120" />
        <el-table-column prop="legalPerson" label="Legal Person" width="120" />
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTag(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="verificationStatus" label="Verification" width="120">
          <template #default="{ row }">
            <el-tag :type="getVerificationTag(row.verificationStatus)">{{ getVerificationText(row.verificationStatus) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="supplierLevel" label="Level" width="70">
          <template #default="{ row }">
            {{ getLevelText(row.supplierLevel) }}
          </template>
        </el-table-column>
        <el-table-column prop="score" label="Score" width="70" />
        <el-table-column prop="createTime" label="Created" width="170">
          <template #default="{ row }">
            {{ row.createTime ? formatDateTime(row.createTime) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="280" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" :icon="View" @click="handleView(row)">View</el-button>
            <el-button v-if="row.verificationStatus === 0" link type="warning" size="small" @click="handleSubmitVerification(row)">Submit Verify</el-button>
            <el-button v-if="row.verificationStatus === 1" link type="success" size="small" @click="handleVerify(row)">Verify</el-button>
            <el-button link type="info" size="small" @click="handleViewPerformance(row)">Performance</el-button>
            <el-button link type="primary" size="small" @click="handleScore(row)">Score</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="isView ? 'Supplier Details' : 'Register Supplier'"
      width="600px"
      :close-on-click-modal="false"
    >
      <el-form v-loading="dialogLoading" :model="formData" label-width="140px">
        <el-form-item label="Supplier Name" prop="supplierName">
          <el-input v-model="formData.supplierName" :disabled="isView" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Contact Person" prop="contactPerson">
              <el-input v-model="formData.contactPerson" :disabled="isView" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Contact Phone" prop="contactPhone">
              <el-input v-model="formData.contactPhone" :disabled="isView" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="Contact Email" prop="contactEmail">
          <el-input v-model="formData.contactEmail" :disabled="isView" />
        </el-form-item>
        <el-form-item label="Address" prop="address">
          <el-input v-model="formData.address" type="textarea" :rows="2" :disabled="isView" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Legal Person" prop="legalPerson">
              <el-input v-model="formData.legalPerson" :disabled="isView" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Business License" prop="businessLicense">
              <el-input v-model="formData.businessLicense" :disabled="isView" />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ isView ? 'Close' : 'Cancel' }}</el-button>
        <el-button v-if="!isView" type="primary" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.mall-supplier-container {
  padding: 20px;
}

.table-card .toolbar {
  margin-bottom: 15px;
}
</style>
