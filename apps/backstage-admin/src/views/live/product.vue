<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus } from '@element-plus/icons-vue'
import {
  getLiveRoomListAPI,
  getLiveProductsByRoomAPI,
  attachProductAPI,
  type LiveRoom,
  type LiveProduct
} from '@/apis/live'

const { t } = useI18n()

const roomId = ref<number | undefined>()
const rooms = ref<LiveRoom[]>([])
const products = ref<LiveProduct[]>([])
const loading = ref(false)

const dialogVisible = ref(false)
const dialogLoading = ref(false)

const formData = ref({
  roomId: '',
  productId: '',
  price: '',
  discountPrice: '',
  stock: ''
})

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
    const response = await getLiveProductsByRoomAPI(roomId.value)
    products.value = response.data || []
    loading.value = false
  } catch (error) {
    loading.value = false
    ElMessage.error('Failed to load products')
  }
}

const handleAdd = () => {
  formData.value = {
    roomId: roomId.value?.toString() || '',
    productId: '',
    price: '',
    discountPrice: '',
    stock: ''
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    await attachProductAPI(formData.value)
    ElMessage.success('Product attached successfully')
    dialogVisible.value = false
    handleSearch()
  } catch (error) {
    ElMessage.error('Failed to attach product')
  } finally {
    dialogLoading.value = false
  }
}

const formatTime = (time: string | undefined) => {
  if (!time) return '-'
  return time.replace('T', ' ').substring(0, 19)
}
</script>

<template>
  <div class="product-container">
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
      <div class="toolbar">
        <el-button type="primary" :icon="Plus" :disabled="roomId === undefined" @click="handleAdd">Attach Product</el-button>
      </div>

      <el-table v-loading="loading" :data="products" border stripe>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="productId" label="Product ID" width="100" />
        <el-table-column prop="productName" label="Product Name" min-width="150" />
        <el-table-column prop="price" label="Price" width="100">
          <template #default="{ row }">
            ${{ (row.price / 100).toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column prop="discountPrice" label="Discount Price" width="120">
          <template #default="{ row }">
            ${{ (row.discountPrice / 100).toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column prop="stock" label="Stock" width="80" />
        <el-table-column prop="soldCount" label="Sold" width="80" />
        <el-table-column prop="createdAt" label="Created" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dialogVisible" title="Attach Product" width="500px" :close-on-click-modal="false">
      <el-form v-loading="dialogLoading" :model="formData" label-width="120px">
        <el-form-item label="Room ID">
          <el-input v-model="formData.roomId" disabled />
        </el-form-item>
        <el-form-item label="Product ID">
          <el-input v-model="formData.productId" placeholder="Product ID" />
        </el-form-item>
        <el-form-item label="Price">
          <el-input v-model="formData.price" placeholder="Price in cents" />
        </el-form-item>
        <el-form-item label="Discount Price">
          <el-input v-model="formData.discountPrice" placeholder="Discount price in cents" />
        </el-form-item>
        <el-form-item label="Stock">
          <el-input v-model="formData.stock" placeholder="Stock quantity" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.product-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card .toolbar {
  margin-bottom: 15px;
}
</style>