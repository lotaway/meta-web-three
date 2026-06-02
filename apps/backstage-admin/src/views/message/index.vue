<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Delete, Refresh } from '@element-plus/icons-vue'
import {
  getNotificationListAPI,
  getNotificationStatisticsAPI,
  createNotificationAPI,
  deleteNotificationAPI,
  batchDeleteNotificationAPI,
  type Notification,
  type NotificationQueryParam
} from '@/apis/message'
import { DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS, MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

// State
const loading = ref(false)
const statistics = ref({
  total: 0,
  unread: 0,
  read: 0
})

// Query params
const query = reactive<NotificationQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  userId: undefined,
  title: '',
  type: '',
  readStatus: undefined,
  startDate: '',
  endDate: ''
})

const notificationList = ref<Notification[]>([])
const notificationTotal = ref(0)

// Dialog state
const dialogVisible = ref(false)
const dialogTitle = ref('')
const notificationForm = reactive({
  userId: undefined as number | undefined,
  title: '',
  content: '',
  type: 'SYSTEM',
  relatedId: '',
  icon: '',
  imageUrl: ''
})
const notificationFormRef = ref()
const editingId = ref<number | undefined>()

const typeOptions = [
  { label: 'System', value: 'SYSTEM' },
  { label: 'Order', value: 'ORDER' },
  { label: 'Payment', value: 'PAYMENT' },
  { label: 'Shipping', value: 'SHIPPING' },
  { label: 'Promotion', value: 'PROMOTION' },
  { label: 'Security', value: 'SECURITY' }
]

const readStatusOptions = [
  { label: 'Unread', value: 0 },
  { label: 'Read', value: 1 }
]

const getReadStatusTagType = (status: number): 'success' | 'warning' | 'info' => {
  return status === 0 ? 'warning' : 'success'
}

const getReadStatusText = (status: number): string => {
  return status === 0 ? 'Unread' : 'Read'
}

const getTypeTagType = (type: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    SYSTEM: 'info',
    ORDER: 'primary',
    PAYMENT: 'success',
    SHIPPING: 'warning',
    PROMOTION: 'danger',
    SECURITY: 'warning'
  }
  return map[type] || 'info'
}

// Methods
const getStatistics = async () => {
  try {
    const res = await getNotificationStatisticsAPI()
    if (res.data) {
      statistics.value = res.data
    }
  } catch (error) {
    console.error('Failed to load statistics:', error)
  }
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getNotificationListAPI(query)
    if (res.data) {
      notificationList.value = res.data.list || []
      notificationTotal.value = res.data.total || 0
    }
  } catch (error) {
    console.error('Failed to load notification list:', error)
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  query.pageNum = 1
  getList()
}

const handleReset = () => {
  query.pageNum = 1
  query.userId = undefined
  query.title = ''
  query.type = ''
  query.readStatus = undefined
  query.startDate = ''
  query.endDate = ''
  getList()
}

const handlePageChange = (page: number) => {
  query.pageNum = page
  getList()
}

const handleSizeChange = (size: number) => {
  query.pageSize = size
  query.pageNum = 1
  getList()
}

const handleRefresh = () => {
  getList()
  getStatistics()
}

const handleCreate = () => {
  dialogTitle.value = t('app.create') + ' Notification'
  editingId.value = undefined
  notificationForm.userId = undefined
  notificationForm.title = ''
  notificationForm.content = ''
  notificationForm.type = 'SYSTEM'
  notificationForm.relatedId = ''
  notificationForm.icon = ''
  notificationForm.imageUrl = ''
  dialogVisible.value = true
}

const handleEdit = (row: Notification) => {
  dialogTitle.value = t('app.edit') + ' Notification'
  editingId.value = row.id
  notificationForm.userId = row.userId
  notificationForm.title = row.title
  notificationForm.content = row.content
  notificationForm.type = row.type
  notificationForm.relatedId = row.relatedId || ''
  notificationForm.icon = row.icon || ''
  notificationForm.imageUrl = row.imageUrl || ''
  dialogVisible.value = true
}

const handleSubmit = async () => {
  if (!notificationForm.title || !notificationForm.content) {
    ElMessage.warning('Please fill in required fields')
    return
  }

  try {
    await createNotificationAPI({
      userId: notificationForm.userId,
      title: notificationForm.title,
      content: notificationForm.content,
      type: notificationForm.type,
      relatedId: notificationForm.relatedId || undefined,
      icon: notificationForm.icon || undefined,
      imageUrl: notificationForm.imageUrl || undefined
    })
    ElMessage.success('Notification created successfully')
    dialogVisible.value = false
    getList()
    getStatistics()
  } catch (error) {
    console.error('Failed to create notification:', error)
  }
}

const handleDelete = async (row: Notification) => {
  try {
    await ElMessageBox.confirm(
      `Are you sure you want to delete this notification?`,
      'Confirm Delete',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    await deleteNotificationAPI(row.id)
    ElMessage.success('Notification deleted successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete notification:', error)
    }
  }
}

const handleBatchDelete = async () => {
  const selectedRows = notificationList.value.filter((item: Notification) => (item as any).selected)
  if (selectedRows.length === 0) {
    ElMessage.warning('Please select at least one notification')
    return
  }

  try {
    await ElMessageBox.confirm(
      `Are you sure you want to delete ${selectedRows.length} notifications?`,
      'Confirm Batch Delete',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    const ids = selectedRows.map((item: Notification) => item.id)
    await batchDeleteNotificationAPI(ids)
    ElMessage.success('Notifications deleted successfully')
    getList()
    getStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to batch delete notifications:', error)
    }
  }
}

const formatDate = (date: string) => {
  if (!date) return '-'
  return new Date(date).toLocaleString()
}

// Lifecycle
onMounted(() => {
  getList()
  getStatistics()
})
</script>

<template>
  <div class="notification-container">
    <el-row :gutter="20" class="statistics-row">
      <el-col :span="8">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label">Total Notifications</div>
            <div class="statistics-value">{{ statistics.total }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label">Unread</div>
            <div class="statistics-value warning">{{ statistics.unread }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" class="statistics-card">
          <div class="statistics-content">
            <div class="statistics-label">Read</div>
            <div class="statistics-value success">{{ statistics.read }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="query-card">
      <el-form :inline="true" :model="query" class="query-form">
        <el-form-item label="User ID">
          <el-input v-model="query.userId" placeholder="User ID" clearable style="width: 150px" />
        </el-form-item>
        <el-form-item label="Title">
          <el-input v-model="query.title" placeholder="Title" clearable style="width: 150px" />
        </el-form-item>
        <el-form-item label="Type">
          <el-select v-model="query.type" placeholder="Select" clearable style="width: 150px">
            <el-option v-for="item in typeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Read Status">
          <el-select v-model="query.readStatus" placeholder="Select" clearable style="width: 120px">
            <el-option v-for="item in readStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Date Range">
          <el-date-picker
            v-model="query.startDate"
            type="date"
            placeholder="Start Date"
            value-format="YYYY-MM-DD"
            style="width: 150px"
          />
          <span style="margin: 0 10px">-</span>
          <el-date-picker
            v-model="query.endDate"
            type="date"
            placeholder="End Date"
            value-format="YYYY-MM-DD"
            style="width: 150px"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleQuery">{{ t('app.search') }}</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <div class="action-bar">
      <el-button type="primary" :icon="Plus" @click="handleCreate">Create Notification</el-button>
      <el-button type="danger" :icon="Delete" @click="handleBatchDelete">Batch Delete</el-button>
      <el-button :icon="Refresh" @click="handleRefresh">Refresh</el-button>
    </div>

    <el-card shadow="never" class="table-card">
      <el-table v-loading="loading" :data="notificationList" stripe style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="userId" label="User ID" width="100" />
        <el-table-column prop="title" label="Title" min-width="150" show-overflow-tooltip />
        <el-table-column prop="content" label="Content" min-width="200" show-overflow-tooltip />
        <el-table-column prop="type" label="Type" width="120">
          <template #default="{ row }">
            <el-tag :type="getTypeTagType(row.type)">{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="readStatus" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getReadStatusTagType(row.readStatus)">{{ getReadStatusText(row.readStatus) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="Create Time" width="180">
          <template #default="{ row }">
            {{ formatDate(row.createTime) }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="120" fixed="right">
          <template #default="{ row }">
            <el-button type="danger" link @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-container">
        <el-pagination
          v-model:current-page="query.pageNum"
          v-model:page-size="query.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="notificationTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>

    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px">
      <el-form ref="notificationFormRef" :model="notificationForm" label-width="120px">
        <el-form-item label="User ID" required>
          <el-input v-model="notificationForm.userId" placeholder="Enter user ID (optional for broadcast)" />
        </el-form-item>
        <el-form-item label="Title" required>
          <el-input v-model="notificationForm.title" placeholder="Enter notification title" />
        </el-form-item>
        <el-form-item label="Content" required>
          <el-input v-model="notificationForm.content" type="textarea" :rows="4" placeholder="Enter notification content" />
        </el-form-item>
        <el-form-item label="Type">
          <el-select v-model="notificationForm.type" style="width: 100%">
            <el-option v-for="item in typeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Related ID">
          <el-input v-model="notificationForm.relatedId" placeholder="Related entity ID (optional)" />
        </el-form-item>
        <el-form-item label="Icon">
          <el-input v-model="notificationForm.icon" placeholder="Icon class or URL (optional)" />
        </el-form-item>
        <el-form-item label="Image URL">
          <el-input v-model="notificationForm.imageUrl" placeholder="Image URL (optional)" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.notification-container {
  padding: 20px;
}

.statistics-row {
  margin-bottom: 20px;
}

.statistics-card {
  text-align: center;
}

.statistics-content {
  padding: 10px;
}

.statistics-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.statistics-value {
  font-size: 28px;
  font-weight: bold;
}

.statistics-value.warning {
  color: #e6a23c;
}

.statistics-value.success {
  color: #67c23a;
}

.query-card {
  margin-bottom: 20px;
}

.query-form :deep(.el-form-item) {
  margin-bottom: 0;
}

.action-bar {
  margin-bottom: 20px;
}

.action-bar .el-button {
  margin-right: 10px;
}

.table-card {
  margin-bottom: 20px;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
</style>