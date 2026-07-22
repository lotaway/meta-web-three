<script setup lang="ts">
import { ref, onMounted, reactive, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listAssets,
  getAsset,
  createAsset,
  updateAsset,
  deleteAsset,
  transferAsset,
  getAssetStatistics,
  type FixedAsset,
  type FixedAssetCreateParams,
  type AssetStatistics
} from '@/apis/asset'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const loading = ref(false)
const statsLoading = ref(false)
const dialogVisible = ref(false)
const transferDialogVisible = ref(false)
const dialogTitle = ref('')
const isEdit = ref(false)
const currentId = ref<number | null>(null)

// Statistics
const statistics = ref<AssetStatistics>({
  totalCount: 0,
  totalOriginalValue: 0,
  totalNetValue: 0,
  totalAccumulatedDepreciation: 0,
  categoryDistribution: [],
  departmentDistribution: [],
  statusDistribution: []
})

// List data
const assetList = ref<FixedAsset[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)

// Search params
const searchParams = reactive({
  status: '',
  category: '',
  departmentId: null as number | null,
  keyword: ''
})

// Form data
const formData = reactive<FixedAssetCreateParams>({
  assetCode: '',
  assetName: '',
  assetCategory: '',
  specification: '',
  model: '',
  serialNumber: '',
  supplierId: undefined,
  supplierName: '',
  manufacturer: '',
  purchaseDate: '',
  originalValue: 0,
  residualValue: 0,
  usefulLife: 0,
  depreciationMethod: 'STRAIGHT_LINE',
  departmentId: undefined,
  departmentName: '',
  location: '',
  custodian: '',
  remark: ''
})

// Transfer form
const transferForm = reactive({
  newDepartmentId: 0,
  newDepartmentName: '',
  newLocation: '',
  newCustodian: ''
})

// Form rules
const rules = {
  assetCode: [{ required: true, message: 'Please enter asset code', trigger: 'blur' }],
  assetName: [{ required: true, message: 'Please enter asset name', trigger: 'blur' }],
  assetCategory: [{ required: true, message: 'Please select asset category', trigger: 'change' }],
  purchaseDate: [{ required: true, message: 'Please select purchase date', trigger: 'change' }],
  originalValue: [{ required: true, message: 'Please enter original value', trigger: 'blur' }],
  usefulLife: [{ required: true, message: 'Please enter useful life', trigger: 'blur' }],
  depreciationMethod: [{ required: true, message: 'Please select depreciation method', trigger: 'change' }]
}

const formRef = ref()
const transferFormRef = ref()

const statusOptions = [
  { value: 'IN_USE', label: 'In Use' },
  { value: 'IDLE', label: 'Idle' },
  { value: 'UNDER_MAINTENANCE', label: 'Under Maintenance' },
  { value: 'DISPOSED', label: 'Disposed' }
]

const categoryOptions = [
  { value: 'ELECTRONIC', label: 'Electronic Equipment' },
  { value: 'FURNITURE', label: 'Furniture' },
  { value: 'VEHICLE', label: 'Vehicle' },
  { value: 'MACHINERY', label: 'Machinery' },
  { value: 'BUILDING', label: 'Building' },
  { value: 'OTHER', label: 'Other' }
]

const depreciationMethodOptions = [
  { value: 'STRAIGHT_LINE', label: 'Straight Line' },
  { value: 'DOUBLE_DECLINING', label: 'Double Declining Balance' },
  { value: 'SUM_OF_YEARS', label: 'Sum of Years Digits' }
]

const formatMoney = (value: number | undefined | null) => {
  if (value === undefined || value === null) return '0.00'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

const getStatusLabel = (status: string) => {
  const map: Record<string, string> = {
    IN_USE: 'In Use',
    IDLE: 'Idle',
    UNDER_MAINTENANCE: 'Under Maintenance',
    DISPOSED: 'Disposed'
  }
  return map[status] || status
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    IN_USE: 'success',
    IDLE: 'info',
    UNDER_MAINTENANCE: 'warning',
    DISPOSED: 'danger'
  }
  return map[status] || 'info'
}

const getDepreciationMethodLabel = (method: string) => {
  const map: Record<string, string> = {
    STRAIGHT_LINE: 'Straight Line',
    DOUBLE_DECLINING: 'Double Declining',
    SUM_OF_YEARS: 'Sum of Years'
  }
  return map[method] || method
}

const loadStatistics = async () => {
  statsLoading.value = true
  try {
    const response = await getAssetStatistics()
    statistics.value = response.data || statistics.value
  } catch (error) {
    console.error('Failed to load statistics:', error)
  } finally {
    statsLoading.value = false
  }
}

const loadData = async () => {
  loading.value = true
  try {
    const params = {
      ...(searchParams.status && { status: searchParams.status }),
      ...(searchParams.category && { category: searchParams.category }),
      ...(searchParams.departmentId && { departmentId: searchParams.departmentId })
    }
    const response = await listAssets(params)
    assetList.value = response.data || []
    total.value = (response.data || []).length
  } catch (error) {
    console.error('Failed to load assets:', error)
    ElMessage.error('Failed to load asset list')
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
  searchParams.category = ''
  searchParams.departmentId = null
  searchParams.keyword = ''
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
  dialogTitle.value = 'Add Asset'
  isEdit.value = false
  currentId.value = null
  resetForm()
  dialogVisible.value = true
}

const handleEdit = async (row: FixedAsset) => {
  dialogTitle.value = 'Edit Asset'
  isEdit.value = true
  currentId.value = row.id
  try {
    const response = await getAsset(row.id)
    Object.assign(formData, response.data)
  } catch (error) {
    console.error('Failed to load asset detail:', error)
    ElMessage.error('Failed to load asset detail')
  }
  dialogVisible.value = true
}

const handleDelete = async (row: FixedAsset) => {
  try {
    await ElMessageBox.confirm(
      `Are you sure you want to delete asset ${row.assetName}?`,
      'Confirm Delete',
      { type: 'warning' }
    )
    await deleteAsset(row.id)
    ElMessage.success('Asset deleted successfully')
    loadData()
    loadStatistics()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to delete asset:', error)
      ElMessage.error('Failed to delete asset')
    }
  }
}

const handleTransfer = (row: FixedAsset) => {
  currentId.value = row.id
  transferForm.newDepartmentId = row.departmentId || 0
  transferForm.newDepartmentName = row.departmentName || ''
  transferForm.newLocation = row.location || ''
  transferForm.newCustodian = row.custodian || ''
  transferDialogVisible.value = true
}

const submitForm = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid: boolean) => {
    if (!valid) return
    try {
      if (isEdit.value && currentId.value) {
        await updateAsset(currentId.value, { ...formData, id: currentId.value })
        ElMessage.success('Asset updated successfully')
      } else {
        await createAsset(formData)
        ElMessage.success('Asset created successfully')
      }
      dialogVisible.value = false
      loadData()
      loadStatistics()
    } catch (error) {
      console.error('Failed to save asset:', error)
      ElMessage.error('Failed to save asset')
    }
  })
}

const submitTransfer = async () => {
  if (!transferFormRef.value) return
  await transferFormRef.value.validate(async (valid: boolean) => {
    if (!valid || !currentId.value) return
    try {
      await transferAsset(currentId.value, transferForm)
      ElMessage.success('Asset transferred successfully')
      transferDialogVisible.value = false
      loadData()
    } catch (error) {
      console.error('Failed to transfer asset:', error)
      ElMessage.error('Failed to transfer asset')
    }
  })
}

const resetForm = () => {
  formData.assetCode = ''
  formData.assetName = ''
  formData.assetCategory = ''
  formData.specification = ''
  formData.model = ''
  formData.serialNumber = ''
  formData.supplierId = undefined
  formData.supplierName = ''
  formData.manufacturer = ''
  formData.purchaseDate = ''
  formData.originalValue = 0
  formData.residualValue = 0
  formData.usefulLife = 0
  formData.depreciationMethod = 'STRAIGHT_LINE'
  formData.departmentId = undefined
  formData.departmentName = ''
  formData.location = ''
  formData.custodian = ''
  formData.remark = ''
}

const cancelDialog = () => {
  dialogVisible.value = false
  resetForm()
}

onMounted(() => {
  loadData()
  loadStatistics()
})
</script>

<template>
  <div class="asset-card-container">
    <!-- Statistics Cards -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Total Assets</div>
            <div class="stat-value">{{ statistics.totalCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Original Value</div>
            <div class="stat-value">¥{{ formatMoney(statistics.totalOriginalValue) }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Net Value</div>
            <div class="stat-value">¥{{ formatMoney(statistics.totalNetValue) }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Accumulated Depreciation</div>
            <div class="stat-value">¥{{ formatMoney(statistics.totalAccumulatedDepreciation) }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Search Bar -->
    <el-card class="search-card">
      <el-form :model="searchParams" inline>
        <el-form-item label="Status">
          <el-select v-model="searchParams.status" placeholder="Select Status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Category">
          <el-select v-model="searchParams.category" placeholder="Select Category" clearable>
            <el-option v-for="item in categoryOptions" :key="item.value" :label="item.label" :value="item.value" />
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
        <el-button type="primary" @click="handleAdd">Add Asset</el-button>
      </div>

      <!-- Table -->
      <el-table :data="assetList" v-loading="loading" stripe>
        <el-table-column prop="assetCode" label="Asset Code" width="120" />
        <el-table-column prop="assetName" label="Asset Name" min-width="150" />
        <el-table-column prop="assetCategory" label="Category" width="130">
          <template #default="{ row }">
            <el-tag>{{ categoryOptions.find(c => c.value === row.assetCategory)?.label || row.assetCategory }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="originalValue" label="Original Value" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.originalValue) }}
          </template>
        </el-table-column>
        <el-table-column prop="accumulatedDepreciation" label="Accumulated" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.accumulatedDepreciation) }}
          </template>
        </el-table-column>
        <el-table-column prop="netValue" label="Net Value" width="130" align="right">
          <template #default="{ row }">
            ¥{{ formatMoney(row.netValue) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="departmentName" label="Department" width="120" />
        <el-table-column prop="location" label="Location" width="120" />
        <el-table-column label="Actions" width="180" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="handleEdit(row)">Edit</el-button>
            <el-button type="warning" link @click="handleTransfer(row)" :disabled="row.status === 'DISPOSED'">Transfer</el-button>
            <el-button type="danger" link @click="handleDelete(row)" :disabled="row.status === 'DISPOSED'">Delete</el-button>
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

    <!-- Add/Edit Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="800px" destroy-on-close>
      <el-form ref="formRef" :model="formData" :rules="rules" label-width="140px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Asset Code" prop="assetCode">
              <el-input v-model="formData.assetCode" placeholder="Enter asset code" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Asset Name" prop="assetName">
              <el-input v-model="formData.assetName" placeholder="Enter asset name" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Category" prop="assetCategory">
              <el-select v-model="formData.assetCategory" placeholder="Select category">
                <el-option v-for="item in categoryOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Specification">
              <el-input v-model="formData.specification" placeholder="Enter specification" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Model">
              <el-input v-model="formData.model" placeholder="Enter model" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Serial Number">
              <el-input v-model="formData.serialNumber" placeholder="Enter serial number" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Purchase Date" prop="purchaseDate">
              <el-date-picker v-model="formData.purchaseDate" type="date" placeholder="Select date" value-format="YYYY-MM-DD" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Original Value" prop="originalValue">
              <el-input-number v-model="formData.originalValue" :min="0" :precision="2" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Residual Value">
              <el-input-number v-model="formData.residualValue" :min="0" :precision="2" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Useful Life (Years)" prop="usefulLife">
              <el-input-number v-model="formData.usefulLife" :min="1" :max="50" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Depreciation Method" prop="depreciationMethod">
              <el-select v-model="formData.depreciationMethod" placeholder="Select method" style="width: 100%">
                <el-option v-for="item in depreciationMethodOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Department Name">
              <el-input v-model="formData.departmentName" placeholder="Enter department name" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Location">
              <el-input v-model="formData.location" placeholder="Enter location" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Custodian">
              <el-input v-model="formData.custodian" placeholder="Enter custodian" />
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

    <!-- Transfer Dialog -->
    <el-dialog v-model="transferDialogVisible" title="Asset Transfer" width="500px" destroy-on-close>
      <el-form ref="transferFormRef" :model="transferForm" label-width="140px">
        <el-form-item label="New Department ID">
          <el-input-number v-model="transferForm.newDepartmentId" :min="0" style="width: 100%" />
        </el-form-item>
        <el-form-item label="New Department Name">
          <el-input v-model="transferForm.newDepartmentName" placeholder="Enter department name" />
        </el-form-item>
        <el-form-item label="New Location">
          <el-input v-model="transferForm.newLocation" placeholder="Enter location" />
        </el-form-item>
        <el-form-item label="New Custodian">
          <el-input v-model="transferForm.newCustodian" placeholder="Enter custodian" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="transferDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="submitTransfer">Confirm</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.asset-card-container {
  padding: 20px;
}

.stats-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
}

.stat-content {
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