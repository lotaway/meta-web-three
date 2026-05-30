<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import {
  getAllSupplierPerformanceAPI,
  getSupplierPerformanceDashboardAPI,
  createOrUpdateSupplierPerformanceAPI,
  deleteSupplierPerformanceAPI,
  type SupplierPerformance,
  type SupplierPerformanceDashboard
} from '@/apis/supplierPerformance'
import { t } from '@/locales'

const list = ref<SupplierPerformance[]>([])
const listLoading = ref(true)
const dashboardData = ref<SupplierPerformanceDashboard | null>(null)
const dashboardLoading = ref(true)

// Dialog
const dialogVisible = ref(false)
const dialogLoading = ref(false)
const isEdit = ref(false)
const formData = ref<SupplierPerformance>({
  supplierId: 0,
  supplierCode: '',
  supplierName: '',
  periodStart: '',
  periodEnd: '',
  onTimeDeliveryRate: 0,
  qualityPassRate: 0,
  priceCompetitivenessScore: 0,
  totalOrders: 0,
  onTimeDeliveryCount: 0,
  qualifiedCount: 0,
  totalQualityCheckCount: 0,
  marketAvgPrice: 0,
  supplierPrice: 0,
  remark: '',
  assessor: ''
})

const assessmentLevelOptions = [
  { value: 'A', label: 'A级 (90分以上)' },
  { value: 'B', label: 'B级 (75-89分)' },
  { value: 'C', label: 'C级 (60-74分)' },
  { value: 'D', label: 'D级 (60分以下)' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getAllSupplierPerformanceAPI()
    listLoading.value = false
    list.value = response.data || []
  } catch (error) {
    listLoading.value = false
    console.error('Failed to load supplier performance:', error)
    ElMessage.error(t('common.loadFailed'))
  }
}

const getDashboard = async () => {
  dashboardLoading.value = true
  try {
    const response = await getSupplierPerformanceDashboardAPI()
    dashboardLoading.value = false
    dashboardData.value = response.data || null
  } catch (error) {
    dashboardLoading.value = false
    console.error('Failed to load dashboard:', error)
  }
}

const handleAdd = () => {
  isEdit.value = false
  formData.value = {
    supplierId: 0,
    supplierCode: '',
    supplierName: '',
    periodStart: '',
    periodEnd: '',
    onTimeDeliveryRate: 0,
    qualityPassRate: 0,
    priceCompetitivenessScore: 0,
    overallScore: 0,
    assessmentLevel: '',
    totalOrders: 0,
    onTimeDeliveryCount: 0,
    qualifiedCount: 0,
    totalQualityCheckCount: 0,
    marketAvgPrice: 0,
    supplierPrice: 0,
    remark: '',
    assessor: '',
    assessmentDate: ''
  } as SupplierPerformance
  dialogVisible.value = true
}

const handleEdit = (row: SupplierPerformance) => {
  isEdit.value = true
  formData.value = { ...row }
  dialogVisible.value = true
}

const handleDelete = async (row: SupplierPerformance) => {
  if (!row.id) return
  try {
    await deleteSupplierPerformanceAPI(row.id)
    ElMessage.success(t('common.deleteSuccess'))
    getList()
    getDashboard()
  } catch (error) {
    console.error('Failed to delete:', error)
    ElMessage.error(t('common.deleteFailed'))
  }
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await createOrUpdateSupplierPerformanceAPI(formData.value)
    ElMessage.success(isEdit.value ? t('common.updateSuccess') : t('common.createSuccess'))
    dialogVisible.value = false
    getList()
    getDashboard()
  } catch (error) {
    console.error('Failed to submit:', error)
    ElMessage.error(isEdit.value ? t('common.updateFailed') : t('common.createFailed'))
  } finally {
    dialogLoading.value = false
  }
}

const getLevelTagType = (level: string) => {
  switch (level) {
    case 'A':
      return 'success'
    case 'B':
      return 'primary'
    case 'C':
      return 'warning'
    case 'D':
      return 'danger'
    default:
      return 'info'
  }
}

const formatScore = (score: number) => {
  if (score === null || score === undefined) return '-'
  return score.toFixed(2)
}

const formatPercent = (value: number) => {
  if (value === null || value === undefined) return '-'
  return value.toFixed(2) + '%'
}

const formatDate = (date: string) => {
  if (!date) return '-'
  return date.split('T')[0]
}

onMounted(() => {
  getList()
  getDashboard()
})
</script>

<template>
  <div class="supplier-performance-container">
    <!-- Dashboard -->
    <el-row :gutter="20" class="dashboard-row" v-loading="dashboardLoading">
      <el-col :span="6">
        <el-card class="dashboard-card">
          <div class="dashboard-title">{{ t('sp.totalSuppliers') }}</div>
          <div class="dashboard-value">{{ dashboardData?.totalSuppliers || 0 }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="dashboard-card dashboard-card-a">
          <div class="dashboard-title">{{ t('sp.levelA') }}</div>
          <div class="dashboard-value">{{ dashboardData?.levelACount || 0 }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="dashboard-card dashboard-card-b">
          <div class="dashboard-title">{{ t('sp.levelB') }}</div>
          <div class="dashboard-value">{{ dashboardData?.levelBCount || 0 }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="dashboard-card dashboard-card-c">
          <div class="dashboard-title">{{ t('sp.levelC') }}</div>
          <div class="dashboard-value">{{ dashboardData?.levelCCount || 0 }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="dashboard-row" v-loading="dashboardLoading">
      <el-col :span="6">
        <el-card class="dashboard-card">
          <div class="dashboard-title">{{ t('sp.avgOnTimeRate') }}</div>
          <div class="dashboard-value">{{ formatPercent(dashboardData?.avgOnTimeDeliveryRate || 0) }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="dashboard-card">
          <div class="dashboard-title">{{ t('sp.avgQualityRate') }}</div>
          <div class="dashboard-value">{{ formatPercent(dashboardData?.avgQualityPassRate || 0) }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="dashboard-card">
          <div class="dashboard-title">{{ t('sp.avgPriceScore') }}</div>
          <div class="dashboard-value">{{ formatScore(dashboardData?.avgPriceCompetitivenessScore || 0) }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="dashboard-card">
          <div class="dashboard-title">{{ t('sp.avgOverallScore') }}</div>
          <div class="dashboard-value">{{ formatScore(dashboardData?.avgOverallScore || 0) }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Action Bar -->
    <div class="action-bar">
      <el-button type="primary" @click="handleAdd">
        {{ t('common.add') }}
      </el-button>
      <el-button @click="getList">
        {{ t('common.refresh') }}
      </el-button>
    </div>

    <!-- Table -->
    <el-table
      v-loading="listLoading"
      :data="list"
      border
      style="width: 100%"
    >
      <el-table-column prop="supplierName" :label="t('sp.supplierName')" min-width="150" />
      <el-table-column prop="supplierCode" :label="t('sp.supplierCode')" width="120" />
      <el-table-column :label="t('sp.period')" width="180">
        <template #default="{ row }">
          {{ formatDate(row.periodStart) }} - {{ formatDate(row.periodEnd) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('sp.onTimeRate')" width="100" align="center">
        <template #default="{ row }">
          {{ formatPercent(row.onTimeDeliveryRate) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('sp.qualityRate')" width="100" align="center">
        <template #default="{ row }">
          {{ formatPercent(row.qualityPassRate) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('sp.priceScore')" width="100" align="center">
        <template #default="{ row }">
          {{ formatScore(row.priceCompetitivenessScore) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('sp.overallScore')" width="100" align="center">
        <template #default="{ row }">
          <strong>{{ formatScore(row.overallScore) }}</strong>
        </template>
      </el-table-column>
      <el-table-column :label="t('sp.level')" width="80" align="center">
        <template #default="{ row }">
          <el-tag :type="getLevelTagType(row.assessmentLevel)">
            {{ row.assessmentLevel }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('sp.assessor')" width="100" prop="assessor" />
      <el-table-column :label="t('sp.assessmentDate')" width="120">
        <template #default="{ row }">
          {{ formatDate(row.assessmentDate) }}
        </template>
      </el-table-column>
      <el-table-column :label="t('common.actions')" width="150" fixed="right">
        <template #default="{ row }">
          <el-button type="primary" link @click="handleEdit(row)">
            {{ t('common.edit') }}
          </el-button>
          <el-button type="danger" link @click="handleDelete(row)">
            {{ t('common.delete') }}
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- Dialog -->
    <el-dialog
      v-model="dialogVisible"
      :title="isEdit ? t('sp.editEvaluation') : t('sp.addEvaluation')"
      width="700px"
      :close-on-click-modal="false"
    >
      <el-form :model="formData" label-width="140px">
        <el-form-item :label="t('sp.supplierName')" required>
          <el-input v-model="formData.supplierName" />
        </el-form-item>
        <el-form-item :label="t('sp.supplierCode')" required>
          <el-input v-model="formData.supplierCode" />
        </el-form-item>
        <el-form-item :label="t('sp.periodStart')" required>
          <el-date-picker
            v-model="formData.periodStart"
            type="date"
            value-format="YYYY-MM-DD"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item :label="t('sp.periodEnd')" required>
          <el-date-picker
            v-model="formData.periodEnd"
            type="date"
            value-format="YYYY-MM-DD"
            style="width: 100%"
          />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('sp.totalOrders')">
              <el-input-number v-model="formData.totalOrders" :min="0" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('sp.onTimeDeliveryCount')">
              <el-input-number v-model="formData.onTimeDeliveryCount" :min="0" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('sp.totalQualityCheckCount')">
              <el-input-number v-model="formData.totalQualityCheckCount" :min="0" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('sp.qualifiedCount')">
              <el-input-number v-model="formData.qualifiedCount" :min="0" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('sp.marketAvgPrice')">
              <el-input-number v-model="formData.marketAvgPrice" :min="0" :precision="2" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('sp.supplierPrice')">
              <el-input-number v-model="formData.supplierPrice" :min="0" :precision="2" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('sp.assessor')">
          <el-input v-model="formData.assessor" />
        </el-form-item>
        <el-form-item :label="t('sp.remark')">
          <el-input v-model="formData.remark" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">
          {{ t('common.submit') }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.supplier-performance-container {
  padding: 20px;
}

.dashboard-row {
  margin-bottom: 20px;
}

.dashboard-card {
  text-align: center;
  
  .dashboard-title {
    font-size: 14px;
    color: #666;
    margin-bottom: 10px;
  }
  
  .dashboard-value {
    font-size: 28px;
    font-weight: bold;
    color: #333;
  }
}

.dashboard-card-a .dashboard-value {
  color: #67c23a;
}

.dashboard-card-b .dashboard-value {
  color: #409eff;
}

.dashboard-card-c .dashboard-value {
  color: #e6a23c;
}

.action-bar {
  margin-bottom: 20px;
  display: flex;
  gap: 10px;
}
</style>