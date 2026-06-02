<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Search, Plus, View, Check, Van, Box, Money } from '@element-plus/icons-vue'
import {
  registerProductAPI,
  getProductInfoAPI,
  getProductListAPI,
  getTraceRecordAPI,
  getTraceEventsAPI,
  createTraceRecordAPI,
  addTraceEventAPI,
  recordProductionAPI,
  recordTransportationAPI,
  recordDeliveryAPI,
  recordSaleAPI,
  verifyProductAPI,
  type ProductInfo,
  type TraceRecord,
  type TraceEvent
} from '@/apis/traceability'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref({ pageNum: 1, pageSize: 10, productId: '', status: '' })
const list = ref<any[]>([])
const listLoading = ref(true)

const productDialogVisible = ref(false)
const traceDialogVisible = ref(false)
const eventDialogVisible = ref(false)
const dialogLoading = ref(false)

const selectedProduct = ref<ProductInfo | null>(null)
const selectedTrace = ref<TraceRecord | null>(null)
const traceEvents = ref<TraceEvent[]>([])

const formData = ref({
  productId: '',
  productName: '',
  batchNumber: '',
  productionDate: '',
  expirationDate: '',
  manufacturer: ''
})

const traceFormData = ref({
  productId: '',
  batchNumber: ''
})

const eventFormData = ref({
  traceId: 0,
  eventType: 'PRODUCTION',
  location: '',
  description: '',
  operator: ''
})

const statusOptions = [
  { label: 'Produced', value: 'PRODUCED' },
  { label: 'In Transit', value: 'IN_TRANSIT' },
  { label: 'Delivered', value: 'DELIVERED' },
  { label: 'Sold', value: 'SOLD' },
  { label: 'Expired', value: 'EXPIRED' }
]

const traceTypeOptions = [
  { label: 'Production', value: 'PRODUCTION' },
  { label: 'Transportation', value: 'TRANSPORTATION' },
  { label: 'Delivery', value: 'DELIVERY' },
  { label: 'Sale', value: 'SALE' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const res = await getProductListAPI({
      pageNum: listQuery.value.pageNum,
      pageSize: listQuery.value.pageSize,
      status: listQuery.value.status || undefined
    })
    list.value = (res.data.list || []).map((item: any) => ({
      id: item.productId,
      productId: item.productId,
      productName: item.productName,
      batchNumber: item.batchNumber || '',
      status: item.isActive ? 'PRODUCED' : 'EXPIRED',
      registeredAt: item.productionDate || new Date().toISOString()
    }))
  } catch (error) {
    ElMessage.error('Failed to load products')
  } finally {
    listLoading.value = false
  }
}

onMounted(() => {
  getList()
})

const handleSearch = () => getList()

const handleReset = () => {
  listQuery.value = { pageNum: 1, pageSize: 10, productId: '', status: '' }
  getList()
}

const handleRegister = async () => {
  if (!formData.value.productId || !formData.value.productName || !formData.value.batchNumber) {
    ElMessage.warning('Please fill in required fields')
    return
  }
  dialogLoading.value = true
  try {
    await registerProductAPI(formData.value)
    ElMessage.success('Product registered successfully')
    productDialogVisible.value = false
    formData.value = { productId: '', productName: '', batchNumber: '', productionDate: '', expirationDate: '', manufacturer: '' }
    getList()
  } catch (error) {
    ElMessage.error('Failed to register product')
  } finally {
    dialogLoading.value = false
  }
}

const handleViewTrace = async (row: any) => {
  try {
    const response = await getProductInfoAPI(row.productId)
    selectedProduct.value = response.data
    const traceResponse = await getTraceRecordAPI(row.id)
    selectedTrace.value = traceResponse.data
    if (traceResponse.data?.id) {
      const eventsResponse = await getTraceEventsAPI(traceResponse.data.id)
      traceEvents.value = eventsResponse.data || []
    }
    traceDialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load trace information')
  }
}

const handleAddEvent = async () => {
  if (!eventFormData.value.location || !eventFormData.value.description) {
    ElMessage.warning('Please fill in required fields')
    return
  }
  dialogLoading.value = true
  try {
    await addTraceEventAPI(eventFormData.value.traceId, {
      eventType: eventFormData.value.eventType,
      location: eventFormData.value.location,
      description: eventFormData.value.description,
      operator: eventFormData.value.operator
    })
    ElMessage.success('Event added successfully')
    eventDialogVisible.value = false
    eventFormData.value = { traceId: 0, eventType: 'PRODUCTION', location: '', description: '', operator: '' }
    if (selectedTrace.value) {
      handleViewTrace({ productId: selectedProduct.value?.productId, id: selectedTrace.value.id })
    }
  } catch (error) {
    ElMessage.error('Failed to add event')
  } finally {
    dialogLoading.value = false
  }
}

const handleVerify = async (row: any) => {
  try {
    const response = await verifyProductAPI(row.productId, row.batchNumber)
    if (response.data?.verified) {
      ElMessage.success('Product verified successfully!')
    } else {
      ElMessage.warning('Product verification failed')
    }
  } catch (error) {
    ElMessage.error('Verification failed')
  }
}

const getStatusLabel = (status: string) => {
  const option = statusOptions.find(s => s.value === status)
  return option?.label || status
}

type StatusTagType = 'info' | 'warning' | 'primary' | 'success' | 'danger'

const getStatusType = (status: string): StatusTagType => {
  const statusMap: Record<string, StatusTagType> = {
    PRODUCED: 'success',
    IN_TRANSIT: 'warning',
    DELIVERED: 'primary',
    SOLD: 'info',
    EXPIRED: 'danger'
  }
  return statusMap[status] || 'info'
}

const getEventIcon = (type: string) => {
  const iconMap: Record<string, any> = {
    PRODUCTION: Box,
    TRANSPORTATION: Van,
    DELIVERY: View,
    SALE: Money
  }
  return iconMap[type] || Box
}
</script>

<template>
  <div class="traceability-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item label="Product ID">
          <el-input v-model="listQuery.productId" placeholder="Enter product ID" clearable />
        </el-form-item>
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="Select status" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
          <el-button type="success" :icon="Plus" @click="productDialogVisible = true">Register Product</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column label="ID" prop="id" width="80" />
        <el-table-column label="Product ID" prop="productId" min-width="100" />
        <el-table-column label="Product Name" prop="productName" min-width="150" />
        <el-table-column label="Batch Number" prop="batchNumber" min-width="130" />
        <el-table-column label="Status" min-width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="Registered At" min-width="160">
          <template #default="{ row }">
            {{ row.registeredAt ? formatDateTime(row.registeredAt) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="200" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" :icon="View" @click="handleViewTrace(row)">
              Trace
            </el-button>
            <el-button type="success" size="small" :icon="Check" @click="handleVerify(row)">
              Verify
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="productDialogVisible" title="Register Product" width="500px">
      <el-form :model="formData" label-width="140px">
        <el-form-item label="Product ID" required>
          <el-input v-model="formData.productId" placeholder="Enter product ID" />
        </el-form-item>
        <el-form-item label="Product Name" required>
          <el-input v-model="formData.productName" placeholder="Enter product name" />
        </el-form-item>
        <el-form-item label="Batch Number" required>
          <el-input v-model="formData.batchNumber" placeholder="Enter batch number" />
        </el-form-item>
        <el-form-item label="Production Date">
          <el-date-picker v-model="formData.productionDate" type="date" placeholder="Select date" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Expiration Date">
          <el-date-picker v-model="formData.expirationDate" type="date" placeholder="Select date" style="width: 100%" />
        </el-form-item>
        <el-form-item label="Manufacturer">
          <el-input v-model="formData.manufacturer" placeholder="Enter manufacturer" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="productDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleRegister">Register</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="traceDialogVisible" title="Product Trace Information" width="700px">
      <div v-if="selectedProduct" class="product-info">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="Product ID">{{ selectedProduct.productId }}</el-descriptions-item>
          <el-descriptions-item label="Product Name">{{ selectedProduct.productName }}</el-descriptions-item>
          <el-descriptions-item label="Batch Number">{{ selectedProduct.batchNumber }}</el-descriptions-item>
          <el-descriptions-item label="Status">
            <el-tag :type="getStatusType(selectedProduct.status)">{{ getStatusLabel(selectedProduct.status) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Manufacturer">{{ selectedProduct.manufacturer || '-' }}</el-descriptions-item>
          <el-descriptions-item label="Production Date">{{ selectedProduct.productionDate || '-' }}</el-descriptions-item>
        </el-descriptions>
      </div>

      <el-divider>Trace Events</el-divider>

      <el-timeline>
        <el-timeline-item
          v-for="event in traceEvents"
          :key="event.id"
          :timestamp="event.timestamp"
          :icon="getEventIcon(event.eventType)"
          :type="event.eventType === 'SALE' ? 'success' : event.eventType === 'PRODUCTION' ? 'primary' : 'warning'"
        >
          <el-card class="event-card">
            <h4>{{ event.eventType }}</h4>
            <p v-if="event.location">Location: {{ event.location }}</p>
            <p v-if="event.description">Description: {{ event.description }}</p>
            <p v-if="event.operator">Operator: {{ event.operator }}</p>
            <p v-if="event.txHash" class="tx-hash">TxHash: {{ event.txHash }}</p>
          </el-card>
        </el-timeline-item>
      </el-timeline>

      <div v-if="traceEvents.length === 0" class="no-events">
        <el-empty description="No trace events found" />
      </div>

      <template #footer>
        <el-button @click="traceDialogVisible = false">Close</el-button>
        <el-button type="primary" @click="eventDialogVisible = true; eventFormData.traceId = selectedTrace?.id || 0">Add Event</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="eventDialogVisible" title="Add Trace Event" width="500px">
      <el-form :model="eventFormData" label-width="120px">
        <el-form-item label="Event Type">
          <el-select v-model="eventFormData.eventType" style="width: 100%">
            <el-option v-for="item in traceTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Location">
          <el-input v-model="eventFormData.location" placeholder="Enter location" />
        </el-form-item>
        <el-form-item label="Description">
          <el-input v-model="eventFormData.description" type="textarea" :rows="3" placeholder="Enter description" />
        </el-form-item>
        <el-form-item label="Operator">
          <el-input v-model="eventFormData.operator" placeholder="Enter operator name" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="eventDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :loading="dialogLoading" @click="handleAddEvent">Add</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.traceability-container {
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

.product-info {
  margin-bottom: 20px;
}

.event-card {
  margin-bottom: 10px;
}

.event-card h4 {
  margin: 0 0 10px 0;
}

.event-card p {
  margin: 5px 0;
  font-size: 14px;
  color: #666;
}

.tx-hash {
  font-family: monospace;
  font-size: 12px;
  color: #409eff;
}

.no-events {
  padding: 40px 0;
}
</style>