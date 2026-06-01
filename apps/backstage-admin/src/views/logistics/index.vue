<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import {
  getLogisticsListAPI,
  updateLogisticsStatusAPI,
  type LogisticsOrder,
  type LogisticsQueryParam
} from '@/apis/logistics'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<LogisticsQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<LogisticsOrder[]>([])
const listLoading = ref(true)
const total = ref(0)

const statusOptions = [
  { label: 'Pending', value: 'PENDING' },
  { label: 'In Transit', value: 'IN_TRANSIT' },
  { label: 'Delivered', value: 'DELIVERED' },
  { label: 'Exception', value: 'EXCEPTION' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getLogisticsListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load logistics orders')
  }
}

onMounted(() => {
  getList()
})

const handleSearch = () => {
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleUpdateStatus = async (row: LogisticsOrder, status: string) => {
  try {
    await updateLogisticsStatusAPI(row.trackingNo, status)
    ElMessage.success('Status updated successfully')
    getList()
  } catch (error) {
    ElMessage.error('Failed to update status')
  }
}

const getStatusLabel = (status: string) => {
  const option = statusOptions.find(s => s.value === status)
  return option?.label || status
}

type StatusTagType = 'primary' | 'success' | 'warning' | 'danger' | 'info'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    PENDING: 'warning',
    IN_TRANSIT: 'primary',
    DELIVERED: 'success',
    EXCEPTION: 'danger'
  }
  return statusMap[status] || 'info'
}
</script>

<template>
  <div class="logistics-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Tracking No">
          <el-input v-model="listQuery.trackingNo" placeholder="Enter tracking number" clearable />
        </el-form-item>
        <el-form-item label="Order No">
          <el-input v-model="listQuery.orderNo" placeholder="Enter order number" clearable />
        </el-form-item>
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option
              v-for="item in statusOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column label="Tracking No" prop="trackingNo" min-width="140" />
        <el-table-column label="Order No" prop="orderNo" min-width="120" />
        <el-table-column label="Carrier" prop="carrierName" min-width="100" />
        <el-table-column label="Status" min-width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Sender" min-width="120">
          <template #default="{ row }">
            <div>{{ row.senderName }}</div>
            <div class="text-xs text-gray-500">{{ row.senderPhone }}</div>
          </template>
        </el-table-column>
        <el-table-column label="Receiver" min-width="120">
          <template #default="{ row }">
            <div>{{ row.receiverName }}</div>
            <div class="text-xs text-gray-500">{{ row.receiverPhone }}</div>
          </template>
        </el-table-column>
        <el-table-column label="Freight Fee" prop="freightFee" min-width="100">
          <template #default="{ row }">
            {{ row.freightFee ? `$${row.freightFee.toFixed(2)}` : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Create Time" min-width="160">
          <template #default="{ row }">
            {{ row.createTime ? formatDateTime(row.createTime) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="180" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="row.status === 'PENDING'"
              type="primary"
              size="small"
              @click="handleUpdateStatus(row, 'IN_TRANSIT')"
            >
              Ship
            </el-button>
            <el-button
              v-if="row.status === 'IN_TRANSIT'"
              type="success"
              size="small"
              @click="handleUpdateStatus(row, 'DELIVERED')"
            >
              Delivered
            </el-button>
            <el-button
              v-if="row.status === 'IN_TRANSIT'"
              type="danger"
              size="small"
              @click="handleUpdateStatus(row, 'EXCEPTION')"
            >
              Exception
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.logistics-container {
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

.text-xs {
  font-size: 12px;
}

.text-gray-500 {
  color: #999;
}
</style>