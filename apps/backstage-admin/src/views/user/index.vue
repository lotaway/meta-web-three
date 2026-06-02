<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, Refresh } from '@element-plus/icons-vue'
import {
  getUserListAPI,
  getUserByIdAPI,
  updateUserAPI,
  updateUserStatusAPI,
  deleteUserAPI,
  deleteUsersBatchAPI,
  getUserStatisticsAPI,
  type User,
  type UserQueryParams,
  type UpdateUserParams,
  type UserStatistics
} from '@/apis/user'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'

// State
const loading = ref(false)
const statistics = ref<UserStatistics>({
  totalUsers: 0,
  activeUsers: 0,
  vipUsers: 0
})

// Query params
const query = reactive<UserQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  keyword: '',
  typeId: undefined,
  email: ''
})

const userList = ref<User[]>([])
const userTotal = ref(0)

// Selected rows
const selectedUsers = ref<User[]>([])
const handleSelectionChange = (rows: User[]) => {
  selectedUsers.value = rows
}

// Dialog state
const dialogVisible = ref(false)
const dialogTitle = ref('')
const userForm = reactive<UpdateUserParams>({
  nickname: '',
  avatar: '',
  email: '',
  telephone: '',
  username: '',
  integration: 0,
  growth: 0,
  memberLevelId: undefined
})
const userFormRef = ref()
const editingId = ref<number | undefined>()

// View dialog
const viewDialogVisible = ref(false)
const viewUser = ref<User | null>(null)

// Status options
const statusOptions = [
  { label: 'Active', value: 1 },
  { label: 'Inactive', value: 0 }
]

const getStatusTagType = (status: number): 'success' | 'info' | 'danger' | 'warning' => {
  const map: Record<string, 'success' | 'info' | 'danger' | 'warning'> = {
    '1': 'success',
    '0': 'danger'
  }
  return map[String(status)] || 'info'
}

const getStatusText = (status: number): string => {
  return status === 1 ? 'Active' : 'Inactive'
}

const getUserTypeText = (typeId: number): string => {
  const typeMap: Record<number, string> = {
    1: 'User',
    2: 'VIP',
    3: 'Partner',
    4: 'Admin'
  }
  return typeMap[typeId] || 'User'
}

// Load statistics
const loadStatistics = async () => {
  try {
    const res = await getUserStatisticsAPI()
    statistics.value = res.data
  } catch (error) {
    console.error('Failed to load statistics:', error)
  }
}

// Load user list
const getUserList = async () => {
  loading.value = true
  try {
    const res = await getUserListAPI(query)
    userList.value = res.data.list
    userTotal.value = res.data.total
  } catch (error) {
    console.error('Failed to load user list:', error)
  } finally {
    loading.value = false
  }
}

// Handle search
const handleSearch = () => {
  query.pageNum = 1
  getUserList()
}

// Handle reset
const handleReset = () => {
  query.keyword = ''
  query.typeId = undefined
  query.email = ''
  query.pageNum = 1
  getUserList()
}

// Handle add
const handleAdd = () => {
  dialogTitle.value = 'Add User'
  editingId.value = undefined
  Object.assign(userForm, {
    nickname: '',
    avatar: '',
    email: '',
    telephone: '',
    username: '',
    integration: 0,
    growth: 0,
    memberLevelId: undefined
  })
  dialogVisible.value = true
}

// Handle edit
const handleEdit = async (row: User) => {
  dialogTitle.value = 'Edit User'
  editingId.value = row.id
  try {
    const res = await getUserByIdAPI(row.id)
    const user = res.data
    Object.assign(userForm, {
      nickname: user.nickname || '',
      avatar: user.avatar || '',
      email: user.email || '',
      telephone: user.telephone || '',
      username: user.username || '',
      integration: user.integration || 0,
      growth: user.growth || 0,
      memberLevelId: user.memberLevelId
    })
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load user details')
  }
}

// Handle view
const handleView = async (row: User) => {
  try {
    const res = await getUserByIdAPI(row.id)
    viewUser.value = res.data
    viewDialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load user details')
  }
}

// Handle submit
const handleSubmit = async () => {
  if (!editingId.value) {
    ElMessage.warning('User creation is not supported in admin panel')
    return
  }
  
  try {
    await updateUserAPI(editingId.value, userForm)
    ElMessage.success('User updated successfully')
    dialogVisible.value = false
    getUserList()
  } catch (error) {
    ElMessage.error('Failed to update user')
  }
}

// Handle status change
const handleStatusChange = async (row: User) => {
  const newStatus = row.status === 1 ? 0 : 1
  try {
    await ElMessageBox.confirm(
      `Are you sure to ${newStatus === 1 ? 'activate' : 'deactivate'} this user?`,
      'Confirm',
      {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    await updateUserStatusAPI(row.id, newStatus)
    ElMessage.success('Status updated successfully')
    getUserList()
  } catch (error) {
    // User cancelled or failed
    getUserList()
  }
}

// Handle delete
const handleDelete = async (row: User) => {
  try {
    await ElMessageBox.confirm(
      'Are you sure to delete this user? This action cannot be undone.',
      'Warning',
      {
        confirmButtonText: 'Delete',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    await deleteUserAPI(row.id)
    ElMessage.success('User deleted successfully')
    getUserList()
  } catch (error) {
    // User cancelled or failed
  }
}

// Handle batch delete
const handleBatchDelete = async () => {
  if (selectedUsers.value.length === 0) {
    ElMessage.warning('Please select at least one user')
    return
  }
  
  try {
    await ElMessageBox.confirm(
      `Are you sure to delete ${selectedUsers.value.length} users? This action cannot be undone.`,
      'Warning',
      {
        confirmButtonText: 'Delete',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }
    )
    const ids = selectedUsers.value.map(u => u.id).join(',')
    await deleteUsersBatchAPI(ids)
    ElMessage.success('Users deleted successfully')
    getUserList()
  } catch (error) {
    // User cancelled or failed
  }
}

// Handle page change
const handlePageChange = (page: number) => {
  query.pageNum = page
  getUserList()
}

// Handle size change
const handleSizeChange = (size: number) => {
  query.pageSize = size
  query.pageNum = 1
  getUserList()
}

// Initial load
onMounted(() => {
  loadStatistics()
  getUserList()
})
</script>

<template>
  <div class="user-management">
    <!-- Statistics -->
    <el-row :gutter="20" class="statistics-row">
      <el-col :span="8">
        <el-card shadow="hover">
          <div class="statistics-card">
            <div class="statistics-value">{{ statistics.totalUsers }}</div>
            <div class="statistics-label">Total Users</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <div class="statistics-card">
            <div class="statistics-value">{{ statistics.activeUsers }}</div>
            <div class="statistics-label">Active Users</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <div class="statistics-card">
            <div class="statistics-value">{{ statistics.vipUsers }}</div>
            <div class="statistics-label">VIP Users</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Search -->
    <el-card class="search-card">
      <el-form :inline="true" :model="query">
        <el-form-item label="Keyword">
          <el-input
            v-model="query.keyword"
            placeholder="Email, nickname or username"
            clearable
            @keyup.enter="handleSearch"
          />
        </el-form-item>
        <el-form-item label="Email">
          <el-input
            v-model="query.email"
            placeholder="Email"
            clearable
            @keyup.enter="handleSearch"
          />
        </el-form-item>
        <el-form-item label="User Type">
          <el-select v-model="query.typeId" placeholder="Select type" clearable>
            <el-option label="User" :value="1" />
            <el-option label="VIP" :value="2" />
            <el-option label="Partner" :value="3" />
            <el-option label="Admin" :value="4" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Actions -->
    <div class="actions-row">
      <el-button type="primary" :icon="Plus" @click="handleAdd">Add User</el-button>
      <el-button
        type="danger"
        :icon="Delete"
        :disabled="selectedUsers.length === 0"
        @click="handleBatchDelete"
      >
        Batch Delete
      </el-button>
    </div>

    <!-- Table -->
    <el-card class="table-card">
      <el-table
        v-loading="loading"
        :data="userList"
        @selection-change="handleSelectionChange"
        stripe
      >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="username" label="Username" min-width="120" />
        <el-table-column prop="nickname" label="Nickname" min-width="120" />
        <el-table-column prop="email" label="Email" min-width="180" />
        <el-table-column prop="telephone" label="Phone" min-width="120" />
        <el-table-column prop="typeId" label="Type" width="100">
          <template #default="{ row }">
            {{ getUserTypeText(row.typeId) }}
          </template>
        </el-table-column>
        <el-table-column prop="integration" label="Integration" width="100" />
        <el-table-column prop="growth" label="Growth" width="100" />
        <el-table-column prop="status" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTagType(row.status)">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="Created At" width="180">
          <template #default="{ row }">
            {{ row.createTime ? row.createTime.slice(0, 19) : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="Actions" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">View</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">Edit</el-button>
            <el-button link type="warning" size="small" @click="handleStatusChange(row)">
              {{ row.status === 1 ? 'Deactivate' : 'Activate' }}
            </el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- Pagination -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="query.pageNum"
          v-model:page-size="query.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="userTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>

    <!-- Edit Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px">
      <el-form ref="userFormRef" :model="userForm" label-width="100px">
        <el-form-item label="Username">
          <el-input v-model="userForm.username" />
        </el-form-item>
        <el-form-item label="Nickname">
          <el-input v-model="userForm.nickname" />
        </el-form-item>
        <el-form-item label="Email">
          <el-input v-model="userForm.email" />
        </el-form-item>
        <el-form-item label="Phone">
          <el-input v-model="userForm.telephone" />
        </el-form-item>
        <el-form-item label="Avatar">
          <el-input v-model="userForm.avatar" placeholder="Avatar URL" />
        </el-form-item>
        <el-form-item label="Integration">
          <el-input-number v-model="userForm.integration" :min="0" />
        </el-form-item>
        <el-form-item label="Growth">
          <el-input-number v-model="userForm.growth" :min="0" />
        </el-form-item>
        <el-form-item label="Member Level">
          <el-input-number v-model="userForm.memberLevelId" :min="0" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleSubmit">Submit</el-button>
      </template>
    </el-dialog>

    <!-- View Dialog -->
    <el-dialog v-model="viewDialogVisible" title="User Details" width="600px">
      <el-descriptions v-if="viewUser" :column="1" border>
        <el-descriptions-item label="ID">{{ viewUser.id }}</el-descriptions-item>
        <el-descriptions-item label="Username">{{ viewUser.username }}</el-descriptions-item>
        <el-descriptions-item label="Nickname">{{ viewUser.nickname }}</el-descriptions-item>
        <el-descriptions-item label="Email">{{ viewUser.email }}</el-descriptions-item>
        <el-descriptions-item label="Phone">{{ viewUser.telephone }}</el-descriptions-item>
        <el-descriptions-item label="Type">{{ getUserTypeText(viewUser.typeId) }}</el-descriptions-item>
        <el-descriptions-item label="Integration">{{ viewUser.integration }}</el-descriptions-item>
        <el-descriptions-item label="Growth">{{ viewUser.growth }}</el-descriptions-item>
        <el-descriptions-item label="Status">
          <el-tag :type="getStatusTagType(viewUser.status)">
            {{ getStatusText(viewUser.status) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Member Level ID">{{ viewUser.memberLevelId }}</el-descriptions-item>
        <el-descriptions-item label="Created At">
          {{ viewUser.createTime ? viewUser.createTime.slice(0, 19) : '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="Updated At">
          {{ viewUser.updateTime ? viewUser.updateTime.slice(0, 19) : '-' }}
        </el-descriptions-item>
      </el-descriptions>
      <template #footer>
        <el-button @click="viewDialogVisible = false">Close</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.user-management {
  padding: 20px;
}

.statistics-row {
  margin-bottom: 20px;
}

.statistics-card {
  text-align: center;
  padding: 20px;
}

.statistics-value {
  font-size: 32px;
  font-weight: bold;
  color: #409eff;
}

.statistics-label {
  font-size: 14px;
  color: #909399;
  margin-top: 10px;
}

.search-card {
  margin-bottom: 20px;
}

.actions-row {
  margin-bottom: 20px;
}

.table-card {
  min-height: 400px;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
</style>