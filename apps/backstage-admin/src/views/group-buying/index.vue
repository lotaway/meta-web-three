<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormRules, TabPaneName } from 'element-plus'
import { Search, Plus, Edit, Delete, Refresh } from '@element-plus/icons-vue'
import {
  getGroupBuyActivityListAPI,
  getGroupBuyStatisticsAPI,
  createGroupBuyActivityAPI,
  updateGroupBuyActivityAPI,
  deleteGroupBuyActivityAPI,
  type GroupBuyActivity,
  type GroupBuyActivityQueryParams,
  type CreateGroupBuyActivityParams,
  type GroupBuyStatistics
} from '@/apis/groupBuying'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

// State
const loading = ref(false)
const activeTab = ref('activities')
const statistics = ref<GroupBuyStatistics>({
  totalActivities: 0,
  activeActivities: 0,
  totalTeams: 0,
  successTeams: 0,
  totalOrders: 0
})

// Query params
const query = reactive<GroupBuyActivityQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  activityName: '',
  productId: undefined,
  status: undefined
})

const activityList = ref<GroupBuyActivity[]>([])
const activityTotal = ref(0)

// Dialog state
const dialogVisible = ref(false)
const dialogTitle = ref('')
const activityForm = reactive<CreateGroupBuyActivityParams>({
  activityName: '',
  productId: 0,
  productName: '',
  singlePrice: 0,
  groupPrice: 0,
  requiredQuantity: 2,
  validityHours: 24,
  startTime: '',
  endTime: '',
  status: 1
})
const activityFormRef = ref()
const editingId = ref<number | undefined>()

const activityRules = reactive<FormRules>({
  activityName: [{ required: true, message: 'Activity name is required', trigger: 'blur' }],
  productId: [{ required: true, message: 'Product ID is required', trigger: 'blur' }],
  productName: [{ required: true, message: 'Product name is required', trigger: 'blur' }],
  singlePrice: [{ required: true, message: 'Single price is required', trigger: 'blur' }],
  groupPrice: [{ required: true, message: 'Group price is required', trigger: 'blur' }],
  requiredQuantity: [{ required: true, message: 'Required quantity is required', trigger: 'blur' }],
  validityHours: [{ required: true, message: 'Validity hours is required', trigger: 'blur' }],
  startTime: [{ required: true, message: 'Start time is required', trigger: 'change' }],
  endTime: [{ required: true, message: 'End time is required', trigger: 'change' }]
})

const statusOptions = [
  { label: 'Active', value: 1 },
  { label: 'Inactive', value: 0 }
]

const getStatusTagType = (status: number): 'success' | 'info' | 'danger' | 'warning' => {
  const map: Record<string, 'success' | 'info' | 'danger' | 'warning'> = {
    '1': 'success',
    '0': 'info'
  }
  return map[String(status)] || 'info'
}

const getStatusText = (status: number): string => {
  return status === 1 ? 'Active' : 'Inactive'
}

// Load statistics
const loadStatistics = async () => {
  try {
    const res = await getGroupBuyStatisticsAPI()
    statistics.value = res.data
  } catch (error) {
    console.error('Failed to load statistics:', error)
  }
}

// Load activity list
const getActivityList = async () => {
  loading.value = true
  try {
    const res = await getGroupBuyActivityListAPI(query)
    activityList.value = res.data.list
    activityTotal.value = res.data.total
  } catch (error) {
    console.error('Failed to load activity list:', error)
  } finally {
    loading.value = false
  }
}

// Handle search
const handleSearch = () => {
  query.pageNum = 1
  getActivityList()
}

// Handle reset
const handleReset = () => {
  query.activityName = ''
  query.productId = undefined
  query.status = undefined
  query.pageNum = 1
  getActivityList()
}

// Handle add
const handleAdd = () => {
  dialogTitle.value = t('groupBuying.addActivity')
  editingId.value = undefined
  Object.assign(activityForm, {
    activityName: '',
    productId: 0,
    productName: '',
    singlePrice: 0,
    groupPrice: 0,
    requiredQuantity: 2,
    validityHours: 24,
    startTime: new Date().toISOString().slice(0, 16),
    endTime: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 16),
    status: 1
  })
  dialogVisible.value = true
}

// Handle edit
const handleEdit = (row: GroupBuyActivity) => {
  dialogTitle.value = t('groupBuying.editActivity')
  editingId.value = row.id
  Object.assign(activityForm, {
    activityName: row.activityName,
    productId: row.productId,
    productName: row.productName,
    singlePrice: row.singlePrice,
    groupPrice: row.groupPrice,
    requiredQuantity: row.requiredQuantity,
    validityHours: row.validityHours,
    startTime: row.startTime ? row.startTime.slice(0, 16) : '',
    endTime: row.endTime ? row.endTime.slice(0, 16) : '',
    status: row.status
  })
  dialogVisible.value = true
}

// Handle submit
const handleSubmit = async () => {
  if (!activityFormRef.value) return

  try {
    await activityFormRef.value.validate()

    const data = {
      ...activityForm,
      startTime: activityForm.startTime,
      endTime: activityForm.endTime
    }

    if (editingId.value) {
      await updateGroupBuyActivityAPI(editingId.value, data)
      ElMessage.success({ message: t('common.updateSuccess'), duration: MESSAGE_DURATION_SHORT })
    } else {
      await createGroupBuyActivityAPI(data)
      ElMessage.success({ message: t('common.createSuccess'), duration: MESSAGE_DURATION_SHORT })
    }

    dialogVisible.value = false
    getActivityList()
    loadStatistics()
  } catch (error) {
    console.error('Failed to submit:', error)
  }
}

// Handle delete
const handleDelete = async (row: GroupBuyActivity) => {
  try {
    await ElMessageBox.confirm(
      t('groupBuying.confirmDelete'),
      t('common.warning'),
      {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
      }
    )

    await deleteGroupBuyActivityAPI(row.id)
    ElMessage.success({ message: t('common.deleteSuccess'), duration: MESSAGE_DURATION_SHORT })
    getActivityList()
    loadStatistics()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete:', error)
    }
  }
}

// Handle refresh
const handleRefresh = () => {
  getActivityList()
  loadStatistics()
}

// Tab change
const handleTabChange = (tab: string | number) => {
  activeTab.value = String(tab)
}

onMounted(() => {
  getActivityList()
  loadStatistics()
})
</script>

<template>
  <div class="group-buying-container">
    <!-- Statistics -->
    <el-row :gutter="20" class="statistics-row">
      <el-col :span="5">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ statistics.totalActivities }}</div>
          <div class="stat-label">{{ t('groupBuying.totalActivities') }}</div>
        </el-card>
      </el-col>
      <el-col :span="5">
        <el-card shadow="hover" class="stat-card active">
          <div class="stat-value">{{ statistics.activeActivities }}</div>
          <div class="stat-label">{{ t('groupBuying.activeActivities') }}</div>
        </el-card>
      </el-col>
      <el-col :span="5">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ statistics.totalTeams }}</div>
          <div class="stat-label">{{ t('groupBuying.totalTeams') }}</div>
        </el-card>
      </el-col>
      <el-col :span="5">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ statistics.successTeams }}</div>
          <div class="stat-label">{{ t('groupBuying.successTeams') }}</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ statistics.totalOrders }}</div>
          <div class="stat-label">{{ t('groupBuying.totalOrders') }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Tabs -->
    <el-tabs v-model="activeTab" @tab-change="handleTabChange">
      <el-tab-pane :label="t('groupBuying.activities')" name="activities">
        <!-- Search -->
        <el-card class="search-card">
          <el-form :inline="true" :model="query">
            <el-form-item :label="t('groupBuying.activityName')">
              <el-input v-model="query.activityName" :placeholder="t('groupBuying.activityNamePlaceholder')" clearable />
            </el-form-item>
            <el-form-item :label="t('groupBuying.status')">
              <el-select v-model="query.status" :placeholder="t('groupBuying.statusPlaceholder')" clearable>
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
              <el-button :icon="Refresh" @click="handleReset">{{ t('common.reset') }}</el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <!-- Table -->
        <el-card class="table-card">
          <div class="table-header">
            <el-button type="primary" :icon="Plus" @click="handleAdd">{{ t('common.add') }}</el-button>
            <el-button :icon="Refresh" @click="handleRefresh">{{ t('common.refresh') }}</el-button>
          </div>
          <el-table v-loading="loading" :data="activityList" stripe>
            <el-table-column prop="id" label="ID" width="80" />
            <el-table-column prop="activityName" :label="t('groupBuying.activityName')" min-width="150" />
            <el-table-column prop="productName" :label="t('groupBuying.productName')" min-width="120" />
            <el-table-column prop="singlePrice" :label="t('groupBuying.singlePrice')" width="100">
              <template #default="{ row }">
                ¥{{ row.singlePrice }}
              </template>
            </el-table-column>
            <el-table-column prop="groupPrice" :label="t('groupBuying.groupPrice')" width="100">
              <template #default="{ row }">
                ¥{{ row.groupPrice }}
              </template>
            </el-table-column>
            <el-table-column prop="requiredQuantity" :label="t('groupBuying.requiredQuantity')" width="120" />
            <el-table-column prop="currentQuantity" :label="t('groupBuying.currentQuantity')" width="120" />
            <el-table-column prop="status" :label="t('groupBuying.status')" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusTagType(row.status)">
                  {{ getStatusText(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="startTime" :label="t('groupBuying.startTime')" width="160">
              <template #default="{ row }">
                {{ row.startTime ? row.startTime.replace('T', ' ') : '-' }}
              </template>
            </el-table-column>
            <el-table-column prop="endTime" :label="t('groupBuying.endTime')" width="160">
              <template #default="{ row }">
                {{ row.endTime ? row.endTime.replace('T', ' ') : '-' }}
              </template>
            </el-table-column>
            <el-table-column :label="t('common.actions')" width="150" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
                <el-button link type="danger" size="small" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-pagination
            v-model:current-page="query.pageNum"
            v-model:page-size="query.pageSize"
            :page-sizes="PAGE_SIZE_OPTIONS"
            :total="activityTotal"
            layout="total, sizes, prev, pager, next, jumper"
            @size-change="getActivityList"
            @current-change="getActivityList"
            class="pagination"
          />
        </el-card>
      </el-tab-pane>

      <el-tab-pane :label="t('groupBuying.teams')" name="teams">
        <el-card>
          <el-empty :description="t('groupBuying.comingSoon')" />
        </el-card>
      </el-tab-pane>

      <el-tab-pane :label="t('groupBuying.orders')" name="orders">
        <el-card>
          <el-empty :description="t('groupBuying.comingSoon')" />
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <!-- Activity Form Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px">
      <el-form ref="activityFormRef" :model="activityForm" label-width="140px" :rules="activityRules">
        <el-form-item :label="t('groupBuying.activityName')" prop="activityName">
          <el-input v-model="activityForm.activityName" :placeholder="t('groupBuying.activityNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.productId')" prop="productId">
          <el-input-number v-model="activityForm.productId" :min="1" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.productName')" prop="productName">
          <el-input v-model="activityForm.productName" :placeholder="t('groupBuying.productNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.singlePrice')" prop="singlePrice">
          <el-input-number v-model="activityForm.singlePrice" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.groupPrice')" prop="groupPrice">
          <el-input-number v-model="activityForm.groupPrice" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.requiredQuantity')" prop="requiredQuantity">
          <el-input-number v-model="activityForm.requiredQuantity" :min="2" :max="100" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.validityHours')" prop="validityHours">
          <el-input-number v-model="activityForm.validityHours" :min="1" :max="168" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.startTime')" prop="startTime">
          <el-date-picker v-model="activityForm.startTime" type="datetime" :placeholder="t('groupBuying.startTimePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.endTime')" prop="endTime">
          <el-date-picker v-model="activityForm.endTime" type="datetime" :placeholder="t('groupBuying.endTimePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('groupBuying.status')" prop="status">
          <el-select v-model="activityForm.status">
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleSubmit">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.group-buying-container {
  padding: 20px;
}

.statistics-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
}

.stat-card.active {
  border-top: 3px solid #67c23a;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card {
  margin-bottom: 20px;
}

.table-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>