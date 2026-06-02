<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import type { ButtonType } from 'element-plus'
import { Search, Plus, Refresh, Edit, Delete, Document } from '@element-plus/icons-vue'
import {
  listAIWarehouseRequestsAPI,
  createAIWarehouseRequestAPI,
  updateAIWarehouseRequestAPI,
  deleteAIWarehouseRequestAPI,
  listAICapabilitiesAPI
} from '@/apis/ai-warehouse'

const { t } = useI18n()

// Types
interface AIWarehouseRequest {
  id?: number
  warehouseId: number
  warehouseName: string
  capabilityType: string
  requestData: string
  responseData?: string
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
  createdAt?: string
}

interface AICapability {
  id: number
  name: string
  type: string
  description: string
  enabled: boolean
}

// Query state
const listQuery = ref({
  pageNum: 1,
  pageSize: 10,
  warehouseId: undefined as number | undefined,
  capabilityType: ''
})
const list = ref<AIWarehouseRequest[]>([])
const listLoading = ref(true)
const total = ref(0)

// Capability state
const capabilityList = ref<AICapability[]>([])
const capabilityLoading = ref(false)

// Dialog state
const dialogVisible = ref(false)
const dialogLoading = ref(false)
const currentRecord = ref<Partial<AIWarehouseRequest>>({
  warehouseId: 1,
  warehouseName: '',
  capabilityType: '',
  requestData: '',
  status: 'PENDING'
})

// Tab state
const activeTab = ref('requests')

// Methods
const handleQuery = async () => {
  listLoading.value = true
  try {
    const res = await listAIWarehouseRequestsAPI({
      pageNum: listQuery.value.pageNum,
      pageSize: listQuery.value.pageSize,
      warehouseId: listQuery.value.warehouseId,
      capabilityType: listQuery.value.capabilityType || undefined
    })
    list.value = res.data?.list || []
    total.value = res.data?.total || 0
  } catch (error) {
    ElMessage.error('Failed to load requests')
  } finally {
    listLoading.value = false
  }
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10,
    warehouseId: undefined,
    capabilityType: ''
  }
  handleQuery()
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageSize = val
  handleQuery()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  handleQuery()
}

const handleAdd = () => {
  currentRecord.value = {
    warehouseId: 1,
    warehouseName: '',
    capabilityType: '',
    requestData: '',
    status: 'PENDING' as const
  }
  dialogVisible.value = true
}

const handleEdit = (row: AIWarehouseRequest) => {
  currentRecord.value = { ...row }
  dialogVisible.value = true
}

const handleDelete = async (row: AIWarehouseRequest) => {
  if (!row.id) return
  try {
    await deleteAIWarehouseRequestAPI(row.id)
    ElMessage.success('Deleted successfully')
    handleQuery()
  } catch (error) {
    ElMessage.error('Failed to delete')
  }
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    if (currentRecord.value.id) {
      await updateAIWarehouseRequestAPI(currentRecord.value.id, currentRecord.value)
      ElMessage.success('Request updated successfully')
    } else {
      await createAIWarehouseRequestAPI(currentRecord.value)
      ElMessage.success('Request submitted successfully')
    }
    dialogVisible.value = false
    handleQuery()
  } catch (error) {
    ElMessage.error('Failed to submit request')
  } finally {
    dialogLoading.value = false
  }
}

const loadCapabilities = async () => {
  capabilityLoading.value = true
  try {
    const res = await listAICapabilitiesAPI()
    capabilityList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load capabilities')
  } finally {
    capabilityLoading.value = false
  }
}

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'warning',
    PROCESSING: 'primary',
    COMPLETED: 'success',
    FAILED: 'danger'
  }
  return map[status] || 'info'
}

onMounted(() => {
  handleQuery()
  loadCapabilities()
})
</script>

<template>
  <div class="warehouse-container">
    <el-card class="header-card">
      <el-tabs v-model="activeTab">
        <el-tab-pane :label="t('warehouse.aiRequests')" name="requests">
          <el-form :inline="true" :model="listQuery" class="search-form">
            <el-form-item :label="t('warehouse.warehouseId')">
              <el-input v-model="listQuery.warehouseId" :placeholder="t('warehouse.warehouseIdPlaceholder')" clearable />
            </el-form-item>
            <el-form-item :label="t('warehouse.capabilityType')">
              <el-select v-model="listQuery.capabilityType" :placeholder="t('warehouse.selectType')" clearable>
                <el-option label="Location Recommendation" value="LOCATION_RECOMMENDATION" />
                <el-option label="Demand Forecasting" value="DEMAND_FORECASTING" />
                <el-option label="Restock Suggestion" value="RESTOCK_SUGGESTION" />
                <el-option label="Anomaly Detection" value="ANOMALY_DETECTION" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleQuery">{{ t('common.query') }}</el-button>
              <el-button :icon="Refresh" @click="handleReset">{{ t('common.reset') }}</el-button>
            </el-form-item>
          </el-form>

          <el-button type="primary" :icon="Plus" @click="handleAdd" class="add-button">
            {{ t('common.add') }}
          </el-button>

          <el-table v-loading="listLoading" :data="list" border stripe>
            <el-table-column type="index" :label="t('common.index')" width="60" />
            <el-table-column prop="warehouseName" :label="t('warehouse.warehouseName')" min-width="120" />
            <el-table-column prop="capabilityType" :label="t('warehouse.capabilityType')" min-width="150" />
            <el-table-column prop="status" :label="t('warehouse.status')" width="120">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="createdAt" :label="t('warehouse.createdAt')" width="180" />
            <el-table-column :label="t('common.actions')" width="150" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" link :icon="Edit" @click="handleEdit(row)">
                  {{ t('common.edit') }}
                </el-button>
                <el-button type="danger" link :icon="Delete" @click="handleDelete(row)">
                  {{ t('common.delete') }}
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
            @size-change="handleSizeChange"
            @current-change="handleCurrentChange"
            class="pagination"
          />
        </el-tab-pane>

        <el-tab-pane :label="t('warehouse.capabilities')" name="capabilities">
          <el-table v-loading="capabilityLoading" :data="capabilityList" border stripe>
            <el-table-column type="index" :label="t('common.index')" width="60" />
            <el-table-column prop="name" :label="t('warehouse.capabilityName')" min-width="150" />
            <el-table-column prop="type" :label="t('warehouse.capabilityType')" min-width="150" />
            <el-table-column prop="description" :label="t('warehouse.description')" min-width="200" />
            <el-table-column prop="enabled" :label="t('warehouse.enabled')" width="100">
              <template #default="{ row }">
                <el-tag :type="row.enabled ? 'success' : 'info'">
                  {{ row.enabled ? t('common.yes') : t('common.no') }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-dialog v-model="dialogVisible" :title="t('warehouse.requestDialog')" width="600px">
      <el-form ref="formRef" :model="currentRecord" label-width="120px">
        <el-form-item :label="t('warehouse.warehouseId')" required>
          <el-input v-model="currentRecord.warehouseId" type="number" />
        </el-form-item>
        <el-form-item :label="t('warehouse.warehouseName')" required>
          <el-input v-model="currentRecord.warehouseName" />
        </el-form-item>
        <el-form-item :label="t('warehouse.capabilityType')" required>
          <el-select v-model="currentRecord.capabilityType">
            <el-option label="Location Recommendation" value="LOCATION_RECOMMENDATION" />
            <el-option label="Demand Forecasting" value="DEMAND_FORECASTING" />
            <el-option label="Restock Suggestion" value="RESTOCK_SUGGESTION" />
            <el-option label="Anomaly Detection" value="ANOMALY_DETECTION" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('warehouse.requestData')">
          <el-input v-model="currentRecord.requestData" type="textarea" :rows="4" />
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

<style scoped>
.warehouse-container {
  padding: 20px;
}

.header-card {
  margin-bottom: 20px;
}

.search-form {
  margin-bottom: 20px;
}

.add-button {
  margin-bottom: 20px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>