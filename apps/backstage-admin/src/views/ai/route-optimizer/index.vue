<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus, Location, Van } from '@element-plus/icons-vue'
import {
  getRoutePlansAPI,
  getVehiclesAPI,
  getAvailableVehiclesAPI,
  createRoutePlanAPI,
  optimizeRouteAPI,
  startRouteAPI,
  completeRouteAPI,
  type RoutePlan,
  type Vehicle
} from '@/apis/routeOptimizer'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref({ pageNum: 1, pageSize: 10, status: '' })
const list = ref<RoutePlan[]>([])
const listLoading = ref(true)
const total = ref(0)

const vehicleList = ref<Vehicle[]>([])
const availableVehicles = ref<Vehicle[]>([])

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const formData = ref({
  planName: '',
  vehicleCode: '',
  optimizationType: 'BALANCED'
})

const statusOptions = [
  { label: 'Draft', value: 'DRAFT' },
  { label: 'Pending', value: 'PENDING' },
  { label: 'Optimizing', value: 'OPTIMIZING' },
  { label: 'In Progress', value: 'IN_PROGRESS' },
  { label: 'Completed', value: 'COMPLETED' },
  { label: 'Cancelled', value: 'CANCELLED' }
]

const optimizationTypeOptions = [
  { label: 'Distance', value: 'DISTANCE' },
  { label: 'Time', value: 'TIME' },
  { label: 'Cost', value: 'COST' },
  { label: 'Balanced', value: 'BALANCED' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getRoutePlansAPI(listQuery.value)
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    ElMessage.error('Failed to load route plans')
  } finally {
    listLoading.value = false
  }
}

const loadVehicles = async () => {
  try {
    const [allVehicles, available] = await Promise.all([
      getVehiclesAPI(),
      getAvailableVehiclesAPI()
    ])
    vehicleList.value = allVehicles.data || []
    availableVehicles.value = available.data || []
  } catch (error) {
    console.error('Failed to load vehicles', error)
  }
}

onMounted(() => {
  getList()
  loadVehicles()
})

const handleSearch = () => getList()

const handleReset = () => {
  listQuery.value = { pageNum: 1, pageSize: 10, status: '' }
  getList()
}

const handleCreate = async () => {
  if (!formData.value.planName || !formData.value.vehicleCode) {
    ElMessage.warning('Please fill in all required fields')
    return
  }
  dialogLoading.value = true
  try {
    await createRoutePlanAPI(formData.value)
    ElMessage.success('Route plan created successfully')
    dialogVisible.value = false
    formData.value = { planName: '', vehicleCode: '', optimizationType: 'BALANCED' }
    getList()
  } catch (error) {
    ElMessage.error('Failed to create route plan')
  } finally {
    dialogLoading.value = false
  }
}

const handleOptimize = async (row: RoutePlan) => {
  try {
    await optimizeRouteAPI(row.id)
    ElMessage.success('Route optimized successfully')
    getList()
  } catch (error) {
    ElMessage.error('Failed to optimize route')
  }
}

const handleStart = async (row: RoutePlan) => {
  try {
    await startRouteAPI(row.id)
    ElMessage.success('Route started successfully')
    getList()
  } catch (error) {
    ElMessage.error('Failed to start route')
  }
}

const handleComplete = async (row: RoutePlan) => {
  try {
    await completeRouteAPI(row.id)
    ElMessage.success('Route completed successfully')
    getList()
  } catch (error) {
    ElMessage.error('Failed to complete route')
  }
}

const getStatusLabel = (status: string) => {
  const option = statusOptions.find(s => s.value === status)
  return option?.label || status
}

const getOptimizationTypeLabel = (type: string) => {
  const option = optimizationTypeOptions.find(o => o.value === type)
  return option?.label || type
}

type StatusTagType = 'info' | 'warning' | 'primary' | 'success' | 'danger'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    DRAFT: 'info',
    PENDING: 'warning',
    OPTIMIZING: 'primary',
    IN_PROGRESS: 'primary',
    COMPLETED: 'success',
    CANCELLED: 'danger'
  }
  return statusMap[status] || 'info'
}
</script>

<template>
  <div class="route-optimizer-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
          <el-button type="success" :icon="Plus" @click="dialogVisible = true">Create Route</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column label="ID" prop="id" width="80" />
        <el-table-column label="Plan Name" prop="planName" min-width="150" />
        <el-table-column label="Vehicle" prop="vehicleCode" min-width="100" />
        <el-table-column label="Optimization" min-width="100">
          <template #default="{ row }">
            {{ getOptimizationTypeLabel(row.optimizationType) }}
          </template>
        </el-table-column>
        <el-table-column label="Status" min-width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Total Distance" prop="totalDistance" min-width="120">
          <template #default="{ row }">
            {{ row.totalDistance ? `${row.totalDistance.toFixed(2)} km` : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Total Cost" prop="totalCost" min-width="100">
          <template #default="{ row }">
            {{ row.totalCost ? `$${row.totalCost.toFixed(2)}` : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Created At" min-width="160">
          <template #default="{ row }">
            {{ row.createdAt ? formatDateTime(row.createdAt) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="280" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="row.status === 'PENDING'"
              type="primary"
              size="small"
              :icon="Location"
              @click="handleOptimize(row)"
            >
              Optimize
            </el-button>
            <el-button
              v-if="row.status === 'OPTIMIZING' || row.status === 'PENDING'"
              type="success"
              size="small"
              @click="handleStart(row)"
            >
              Start
            </el-button>
            <el-button
              v-if="row.status === 'IN_PROGRESS'"
              type="warning"
              size="small"
              @click="handleComplete(row)"
            >
              Complete
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dialogVisible" title="Create Route Plan" width="500px">
      <el-form :model="formData" label-width="120px">
        <el-form-item label="Plan Name" required>
          <el-input v-model="formData.planName" placeholder="Enter plan name" />
        </el-form-item>
        <el-form-item label="Vehicle" required>
          <el-select v-model="formData.vehicleCode" placeholder="Select vehicle" style="width: 100%">
            <el-option
              v-for="vehicle in availableVehicles"
              :key="vehicle.vehicleCode"
              :label="`${vehicle.vehicleCode} - ${vehicle.vehicleNumber}`"
              :value="vehicle.vehicleCode"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Optimization Type">
          <el-select v-model="formData.optimizationType" style="width: 100%">
            <el-option
              v-for="item in optimizationTypeOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleCreate">Create</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.route-optimizer-container {
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