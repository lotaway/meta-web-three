<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus, Refresh, Check, Edit, Document } from '@element-plus/icons-vue'
import {
  getForecastByWarehouseAPI,
  getAllModelsAPI,
  confirmForecastAPI,
  adjustForecastAPI,
  recordActualSalesAPI,
  createModelAPI,
  trainModelAPI,
  deployModelAPI,
  type Forecast,
  type ForecastModel,
  type ForecastQueryParam
} from '@/apis/forecasting'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<ForecastQueryParam>({ pageNum: 1, pageSize: 10 })
const list = ref<Forecast[]>([])
const listLoading = ref(true)
const total = ref(0)

const modelList = ref<ForecastModel[]>([])
const modelLoading = ref(false)

const activeTab = ref('forecast')
const dialogVisible = ref(false)
const dialogLoading = ref(false)
const modelDialogVisible = ref(false)
const modelFormLoading = ref(false)

const formData = ref({
  skuCode: '',
  skuName: '',
  warehouseId: 1,
  forecastDate: '',
  quantity: 100,
  modelName: ''
})

const modelFormData = ref({
  modelName: '',
  modelType: 'DEMAND_FORECAST',
  algorithm: 'ARIMA',
  trainingDays: 30
})

const statusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'Confirmed', value: 'CONFIRMED' },
  { label: 'Adjusted', value: 'ADJUSTED' },
  { label: 'Actual Recorded', value: 'ACTUAL_RECORDED' }
]

const modelStatusOptions = [
  { label: 'Draft', value: 'DRAFT' },
  { label: 'Training', value: 'TRAINING' },
  { label: 'Trained', value: 'TRAINED' },
  { label: 'Deployed', value: 'DEPLOYED' },
  { label: 'Archived', value: 'ARCHIVED' }
]

const algorithmOptions = [
  { label: 'ARIMA', value: 'ARIMA' },
  { label: 'Prophet', value: 'PROPHET' },
  { label: 'LSTM', value: 'LSTM' },
  { label: 'XGBoost', value: 'XGBOOST' }
]

const adjustDialogVisible = ref(false)
const adjustForm = ref({ id: 0, newQuantity: 0 })

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getForecastByWarehouseAPI(1)
    list.value = response.data || []
    total.value = list.value.length
  } catch (error) {
    ElMessage.error('Failed to load forecasts')
  } finally {
    listLoading.value = false
  }
}

const getModels = async () => {
  modelLoading.value = true
  try {
    const response = await getAllModelsAPI()
    modelList.value = response.data || []
  } catch (error) {
    ElMessage.error('Failed to load models')
  } finally {
    modelLoading.value = false
  }
}

onMounted(() => {
  getList()
  getModels()
})

const handleSearch = () => getList()

const handleReset = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
  getList()
}

const handleCreate = async () => {
  if (!formData.value.skuCode || !formData.value.forecastDate) {
    ElMessage.warning('Please fill in required fields')
    return
  }
  dialogLoading.value = true
  try {
    await createForecastAPI(formData.value)
    ElMessage.success('Forecast created successfully')
    dialogVisible.value = false
    formData.value = { skuCode: '', skuName: '', warehouseId: 1, forecastDate: '', quantity: 100, modelName: '' }
    getList()
  } catch (error) {
    ElMessage.error('Failed to create forecast')
  } finally {
    dialogLoading.value = false
  }
}

const handleConfirm = async (row: Forecast) => {
  try {
    await confirmForecastAPI(row.id)
    ElMessage.success('Forecast confirmed')
    getList()
  } catch (error) {
    ElMessage.error('Failed to confirm forecast')
  }
}

const handleAdjust = (row: Forecast) => {
  adjustForm.value = { id: row.id, newQuantity: row.predictedQuantity }
  adjustDialogVisible.value = true
}

const submitAdjust = async () => {
  try {
    await adjustForecastAPI(adjustForm.value.id, adjustForm.value.newQuantity)
    ElMessage.success('Forecast adjusted')
    adjustDialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error('Failed to adjust forecast')
  }
}

const handleRecordActual = async (row: Forecast) => {
  try {
    await recordActualSalesAPI(row.id, row.actualQuantity || 0)
    ElMessage.success('Actual sales recorded')
    getList()
  } catch (error) {
    ElMessage.error('Failed to record actual sales')
  }
}

const handleCreateModel = async () => {
  if (!modelFormData.value.modelName) {
    ElMessage.warning('Please enter model name')
    return
  }
  modelFormLoading.value = true
  try {
    await createModelAPI(modelFormData.value)
    ElMessage.success('Model created successfully')
    modelDialogVisible.value = false
    modelFormData.value = { modelName: '', modelType: 'DEMAND_FORECAST', algorithm: 'ARIMA', trainingDays: 30 }
    getModels()
  } catch (error) {
    ElMessage.error('Failed to create model')
  } finally {
    modelFormLoading.value = false
  }
}

const handleTrainModel = async (row: ForecastModel) => {
  try {
    await trainModelAPI(row.id)
    ElMessage.success('Model training started')
    getModels()
  } catch (error) {
    ElMessage.error('Failed to start training')
  }
}

const handleDeployModel = async (row: ForecastModel) => {
  try {
    await deployModelAPI(row.id)
    ElMessage.success('Model deployed successfully')
    getModels()
  } catch (error) {
    ElMessage.error('Failed to deploy model')
  }
}

const getStatusLabel = (status: string) => {
  const option = statusOptions.find(s => s.value === status)
  return option?.label || status
}

const getModelStatusLabel = (status: string) => {
  const option = modelStatusOptions.find(s => s.value === status)
  return option?.label || status
}

type StatusTagType = 'info' | 'warning' | 'primary' | 'success' | 'danger'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    PENDING: 'warning',
    CONFIRMED: 'success',
    ADJUSTED: 'primary',
    ACTUAL_RECORDED: 'info',
    DRAFT: 'info',
    TRAINING: 'warning',
    TRAINED: 'primary',
    DEPLOYED: 'success',
    ARCHIVED: 'info'
  }
  return statusMap[status] || 'info'
}

const createForecastAPI = async (data: any) => {
  const http = (await import('@/utils/http')).default
  return http({ url: '/api/forecasting/forecast', method: 'post', data })
}
</script>

<template>
  <div class="forecasting-container">
    <el-tabs v-model="activeTab" type="border-card">
      <el-tab-pane label="Forecast Management" name="forecast">
        <el-card class="search-card">
          <el-form :inline="true" :model="listQuery" class="search-form">
            <el-form-item label="SKU Code">
              <el-input v-model="listQuery.skuCode" placeholder="Enter SKU code" clearable />
            </el-form-item>
            <el-form-item label="Status">
              <el-select v-model="listQuery.status" placeholder="Select status" clearable>
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
              <el-button @click="handleReset">Reset</el-button>
              <el-button type="success" :icon="Plus" @click="dialogVisible = true">Create Forecast</el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <el-card class="table-card">
          <el-table v-loading="listLoading" :data="list" border stripe>
            <el-table-column label="ID" prop="id" width="80" />
            <el-table-column label="SKU Code" prop="skuCode" min-width="120" />
            <el-table-column label="SKU Name" prop="skuName" min-width="150" />
            <el-table-column label="Forecast Date" min-width="120">
              <template #default="{ row }">
                {{ row.forecastDate ? row.forecastDate.split('T')[0] : '-' }}
              </template>
            </el-table-column>
            <el-table-column label="Predicted Qty" prop="predictedQuantity" min-width="120" />
            <el-table-column label="Actual Qty" prop="actualQuantity" min-width="100">
              <template #default="{ row }">
                {{ row.actualQuantity || '-' }}
              </template>
            </el-table-column>
            <el-table-column label="Accuracy" prop="accuracy" min-width="100">
              <template #default="{ row }">
                {{ row.accuracy ? `${row.accuracy.toFixed(1)}%` : '-' }}
              </template>
            </el-table-column>
            <el-table-column label="Status" min-width="120">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">
                  {{ getStatusLabel(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="Created At" min-width="160">
              <template #default="{ row }">
                {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="200" fixed="right">
              <template #default="{ row }">
                <el-button v-if="row.status === 'PENDING'" type="success" size="small" :icon="Check" @click="handleConfirm(row)">
                  Confirm
                </el-button>
                <el-button v-if="row.status === 'CONFIRMED' || row.status === 'PENDING'" type="primary" size="small" :icon="Edit" @click="handleAdjust(row)">
                  Adjust
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Model Management" name="model">
        <el-card class="search-card">
          <el-form :inline="true">
            <el-form-item>
              <el-button type="success" :icon="Plus" @click="modelDialogVisible = true">Create Model</el-button>
              <el-button :icon="Refresh" @click="getModels">Refresh</el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <el-card class="table-card">
          <el-table v-loading="modelLoading" :data="modelList" border stripe>
            <el-table-column label="ID" prop="id" width="80" />
            <el-table-column label="Model Name" prop="modelName" min-width="150" />
            <el-table-column label="Model Type" prop="modelType" min-width="120" />
            <el-table-column label="Algorithm" prop="algorithm" min-width="100" />
            <el-table-column label="Training Days" prop="trainingDays" min-width="120" />
            <el-table-column label="Status" min-width="120">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">
                  {{ getModelStatusLabel(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="Created At" min-width="160">
              <template #default="{ row }">
                {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="200" fixed="right">
              <template #default="{ row }">
                <el-button v-if="row.status === 'TRAINED'" type="success" size="small" @click="handleDeployModel(row)">
                  Deploy
                </el-button>
                <el-button v-if="row.status === 'DRAFT' || row.status === 'ARCHIVED'" type="primary" size="small" @click="handleTrainModel(row)">
                  Train
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="dialogVisible" title="Create Forecast" width="500px">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="SKU Code" required>
          <el-input v-model="formData.skuCode" placeholder="Enter SKU code" />
        </el-form-item>
        <el-form-item label="SKU Name">
          <el-input v-model="formData.skuName" placeholder="Enter SKU name" />
        </el-form-item>
        <el-form-item label="Forecast Date" required>
          <el-date-picker v-model="formData.forecastDate" type="date" placeholder="Select date" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Quantity">
          <el-input-number v-model="formData.quantity" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Model Name">
          <el-input v-model="formData.modelName" placeholder="Optional model name" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleCreate">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="modelDialogVisible" title="Create Forecast Model" width="500px">
      <el-form :model="modelFormData" label-width="120px">
        <el-form-item label="Model Name" required>
          <el-input v-model="modelFormData.modelName" placeholder="Enter model name" />
        </el-form-item>
        <el-form-item label="Model Type">
          <el-select v-model="modelFormData.modelType" style="width: 100%">
            <el-option label="Demand Forecast" value="DEMAND_FORECAST" />
            <el-option label="Sales Forecast" value="SALES_FORECAST" />
          </el-select>
        </el-form-item>
        <el-form-item label="Algorithm">
          <el-select v-model="modelFormData.algorithm" style="width: 100%">
            <el-option v-for="item in algorithmOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Training Days">
          <el-input-number v-model="modelFormData.trainingDays" :min="1" :max="365" style="width: 100%" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="modelDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="modelFormLoading" @click="handleCreateModel">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="adjustDialogVisible" title="Adjust Forecast" width="400px">
      <el-form label-width="120px">
        <el-form-item label="New Quantity">
          <el-input-number v-model="adjustForm.newQuantity" :min="0" style="width: 100%" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="adjustDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="submitAdjust">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.forecasting-container {
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