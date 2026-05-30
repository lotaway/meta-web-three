<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listInventory,
  createInventory,
  confirmInventory,
  getInventoryStatistics,
  listAssets,
  type FixedAssetInventory,
  type AssetInventoryCreateParams,
  type InventoryStatistics,
  type FixedAsset
} from '@/apis/asset'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const loading = ref(false)
const statsLoading = ref(false)
const dialogVisible = ref(false)
const dialogTitle = ref('Add Inventory')
const isEdit = ref(false)
const currentId = ref<number | null>(null)

// Statistics
const statistics = ref<InventoryStatistics>({
  totalCount: 0,
  pendingCount: 0,
  completedCount: 0,
  discrepancyCount: 0,
  resultDistribution: []
})

// List data
const inventoryList = ref<FixedAssetInventory[]>([])
const assetList = ref<FixedAsset[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)

// Search params
const searchParams = reactive({
  status: ''
})

// Form data
const initialFormData = (): Required<AssetInventoryCreateParams> => ({
  inventoryCode: '',
  inventoryName: '',
  inventoryDate: new Date().toISOString().split('T')[0] as string,
  departmentId: 0,
  departmentName: '',
  inventoryPerson: '',
  assetId: 0,
  assetCode: '',
  bookLocation: '',
  actualLocation: '',
  inventoryResult: '',
  discrepancyReason: '',
  handleMethod: '',
  remark: '',
  createdBy: 0,
  creatorName: ''
})

const formData = reactive(initialFormData()) as any

const formRef = ref()

const statusOptions = [
  { value: 'PENDING', label: 'Pending' },
  { value: 'COMPLETED', label: 'Completed' },
  { value: 'DISCREPANCY', label: 'Discrepancy' }
]

const resultOptions = [
  { value: 'MATCH', label: 'Match' },
  { value: 'MISSING', label: 'Missing' },
  { value: 'DAMAGED', label: 'Damaged' },
  { value: 'EXTRA', label: 'Extra' }
]

const handleMethodOptions = [
  { value: 'REPLENISH', label: 'Replenish' },
  { value: 'WRITE_OFF', label: 'Write Off' },
  { value: 'ADJUST', label: 'Adjust' },
  { value: 'FOLLOW_UP', label: 'Follow Up' }
]

const formatDate = (date: string) => {
  if (!date) return '-'
  return date.split('T')[0]
}

const getStatusLabel = (status: string) => {
  const map: Record<string, string> = {
    PENDING: 'Pending',
    COMPLETED: 'Completed',
    DISCREPANCY: 'Discrepancy'
  }
  return map[status] || status
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'warning',
    COMPLETED: 'success',
    DISCREPANCY: 'danger'
  }
  return map[status] || 'info'
}

const getResultLabel = (result: string) => {
  const map: Record<string, string> = {
    MATCH: 'Match',
    MISSING: 'Missing',
    DAMAGED: 'Damaged',
    EXTRA: 'Extra'
  }
  return map[result] || result
}

const loadStatistics = async () => {
  statsLoading.value = true
  try {
    const response = await getInventoryStatistics()
    statistics.value = response.data || statistics.value
  } catch (error) {
    console.error('Failed to load statistics:', error)
  } finally {
    statsLoading.value = false
  }
}

const loadAssets = async () => {
  try {
    const response = await listAssets()
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
    const response = await listInventory(params)
    inventoryList.value = response.data || []
    total.value = (response.data || []).length
  } catch (error) {
    console.error('Failed to load inventory list:', error)
    ElMessage.error('Failed to load inventory list')
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
  dialogTitle.value = 'Add Inventory'
  isEdit.value = false
  currentId.value = null
  resetForm()
  dialogVisible.value = true
}

const handleAssetSelect = (assetId: number) => {
  const asset = assetList.value.find(a => a.id === assetId)
  if (asset) {
    formData.assetCode = asset.assetCode || ''
    formData.bookLocation = asset.location || ''
  }
}

const handleConfirm = async (row: FixedAssetInventory) => {
  try {
    const { value: handleResult } = await ElMessageBox.prompt(
      'Please enter handle result',
      'Confirm Inventory',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        inputPattern: /.+/,
        inputErrorMessage: 'Handle result is required'
      }
    )
    await confirmInventory(row.id, { handleResult })
    ElMessage.success('Inventory confirmed successfully')
    loadData()
    loadStatistics()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('Failed to confirm inventory:', error)
      ElMessage.error('Failed to confirm inventory')
    }
  }
}

const submitForm = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid: boolean) => {
    if (!valid) return
    try {
      await createInventory({ ...formData })
      ElMessage.success('Inventory created successfully')
      dialogVisible.value = false
      loadData()
      loadStatistics()
    } catch (error) {
      console.error('Failed to create inventory:', error)
      ElMessage.error('Failed to create inventory')
    }
  })
}

const resetForm = () => {
  const data = initialFormData()
  Object.assign(formData, data)
}

const cancelDialog = () => {
  dialogVisible.value = false
  resetForm()
}

onMounted(() => {
  loadData()
  loadStatistics()
  loadAssets()
})
</script>

<template>
  <div class="inventory-container">
    <!-- Statistics Cards -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Total</div>
            <div class="stat-value">{{ statistics.totalCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Pending</div>
            <div class="stat-value" style="color: #E6A23C">{{ statistics.pendingCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Completed</div>
            <div class="stat-value" style="color: #67C23A">{{ statistics.completedCount }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <div class="stat-label">Discrepancy</div>
            <div class="stat-value" style="color: #F56C6C">{{ statistics.discrepancyCount }}</div>
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
        <el-form-item>
          <el-button type="primary" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Action Bar -->
    <el-card class="table-card">
      <div class="table-header">
        <el-button type="primary" @click="handleAdd">Add Inventory</el-button>
      </div>

      <!-- Table -->
      <el-table :data="inventoryList" v-loading="loading" stripe>
        <el-table-column prop="inventoryCode" label="Inventory Code" width="140" />
        <el-table-column prop="inventoryName" label="Inventory Name" min-width="150" />
        <el-table-column prop="inventoryDate" label="Date" width="100">
          <template #default="{ row }">
            {{ formatDate(row.inventoryDate) }}
          </template>
        </el-table-column>
        <el-table-column prop="departmentName" label="Department" width="120" />
        <el-table-column prop="inventoryPerson" label="Person" width="100" />
        <el-table-column prop="assetCode" label="Asset Code" width="120" />
        <el-table-column prop="bookLocation" label="Book Location" width="120" />
        <el-table-column prop="actualLocation" label="Actual Location" width="120" />
        <el-table-column prop="inventoryResult" label="Result" width="100">
          <template #default="{ row }">
            <el-tag v-if="row.inventoryResult" :type="row.inventoryResult === 'MATCH' ? 'success' : 'danger'">
              {{ getResultLabel(row.inventoryResult) }}
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="120" fixed="right">
          <template #default="{ row }">
            <el-button type="success" link @click="handleConfirm(row)" :disabled="row.status === 'COMPLETED'">
              Confirm
            </el-button>
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
            <el-form-item label="Inventory Code">
              <el-input v-model="formData.inventoryCode" placeholder="Enter code" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Inventory Name">
              <el-input v-model="formData.inventoryName" placeholder="Enter name" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Inventory Date">
              <el-date-picker v-model="formData.inventoryDate" type="date" placeholder="Select date" value-format="YYYY-MM-DD" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Department Name">
              <el-input v-model="formData.departmentName" placeholder="Enter department" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Inventory Person">
              <el-input v-model="formData.inventoryPerson" placeholder="Enter person" />
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
            <el-form-item label="Book Location">
              <el-input v-model="formData.bookLocation" placeholder="Book location" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Actual Location">
              <el-input v-model="formData.actualLocation" placeholder="Actual location" />
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
  </div>
</template>

<style scoped>
.inventory-container {
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
