<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import {
  getLiveRoomListAPI,
  getLiveOrdersByRoomAPI,
  type LiveRoom,
  type LiveOrder
} from '@/apis/live'

const { t } = useI18n()

const roomId = ref<number | undefined>()
const rooms = ref<LiveRoom[]>([])
const orders = ref<LiveOrder[]>([])
const loading = ref(false)

const getRooms = async () => {
  try {
    const response = await getLiveRoomListAPI({ pageNum: 1, pageSize: 100 })
    rooms.value = (response as any).data || []
  } catch (error) {
    ElMessage.error('Failed to load rooms')
  }
}

const handleSearch = async () => {
  if (!roomId.value) {
    ElMessage.warning('Please select a room')
    return
  }
  loading.value = true
  try {
    const response = await getLiveOrdersByRoomAPI(roomId.value)
    orders.value = response.data || []
    loading.value = false
  } catch (error) {
    loading.value = false
    ElMessage.error('Failed to load orders')
  }
}

const getStatusType = (status: string): 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    'PENDING': 'warning',
    'PAID': 'success',
    'SHIPPED': 'info',
    'COMPLETED': 'success',
    'CANCELLED': 'danger'
  }
  return map[status]
}

const formatTime = (time: string | undefined) => {
  if (!time) return '-'
  return time.replace('T', ' ').substring(0, 19)
}

onMounted(() => {
  getRooms()
})
</script>

<template>
  <div class="order-container">
    <el-card class="search-card">
      <el-form :inline="true" class="search-form">
        <el-form-item label="Live Room">
          <el-select v-model="roomId" placeholder="Select room" @change="handleSearch">
            <el-option v-for="room in rooms" :key="room.id || 0" :label="room.roomName" :value="room.id || 0" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="loading" :data="orders" border stripe>
        <el-table-column prop="id" label="Order ID" width="100" />
        <el-table-column prop="productId" label="Product ID" width="100" />
        <el-table-column prop="productName" label="Product Name" min-width="150" />
        <el-table-column prop="userId" label="User ID" width="100" />
        <el-table-column prop="userName" label="Username" width="120" />
        <el-table-column prop="quantity" label="Quantity" width="80" />
        <el-table-column prop="totalAmount" label="Amount" width="100">
          <template #default="{ row }">
            ${{ (row.totalAmount / 100).toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="Created" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.order-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}
</style>