<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh, Delete } from '@element-plus/icons-vue'
import {
  getUserStorageListAPI,
  getUserStorageByIdAPI,
  deleteUserStorageAPI,
  getMediaStatisticsAPI,
  getQuotaConfigAPI,
  type UserStorage,
  type UserStorageQueryParam,
  type MediaStatistics,
  type QuotaConfigResponse,
} from '@/apis/media'
import { DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'

const loading = ref(false)
const list = ref<UserStorage[]>([])
const total = ref(0)
const statistics = ref<MediaStatistics | null>(null)
const quotaConfig = ref<QuotaConfigResponse | null>(null)

const query = reactive<UserStorageQueryParam>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
})

const loadStatistics = async () => {
  try {
    const res = await getMediaStatisticsAPI()
    statistics.value = res.data
  } catch (error) { console.error(error) }
}

const loadConfig = async () => {
  try {
    const res = await getQuotaConfigAPI()
    quotaConfig.value = res.data
  } catch (error) { console.error(error) }
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getUserStorageListAPI(query)
    list.value = res.data.list
    total.value = res.data.total
  } catch (error) { console.error(error) }
  finally { loading.value = false }
}

onMounted(() => { getList(); loadStatistics(); loadConfig() })

const handleSearch = () => { query.pageNum = 1; getList() }
const handleReset = () => { query.userId = undefined; query.pageNum = 1; getList() }

const handleDelete = async (row: UserStorage) => {
  try {
    await ElMessageBox.confirm('Delete storage record?', 'Confirm', {
      confirmButtonText: 'Confirm', cancelButtonText: 'Cancel', type: 'warning',
    })
    await deleteUserStorageAPI(row.id)
    ElMessage.success('Deleted')
    getList()
    loadStatistics()
  } catch (error) {
    if (error !== 'cancel') ElMessage.error('Failed to delete')
  }
}

const formatBytes = (bytes: number): string => {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let val = bytes
  while (val >= 1024 && i < units.length - 1) { val /= 1024; i++ }
  return `${val.toFixed(2)} ${units[i]}`
}
</script>

<template>
  <div class="media-container">
    <el-row :gutter="20" class="stat-row">
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ statistics?.totalUsers || 0 }}</div>
          <div class="stat-label">Total Users</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ statistics ? formatBytes(statistics.totalUsed) : '-' }}</div>
          <div class="stat-label">Total Storage Used</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ statistics ? formatBytes(statistics.averageUsed) : '-' }}</div>
          <div class="stat-label">Avg Per User</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-value">{{ statistics ? formatBytes(statistics.maxUsed) : '-' }}</div>
          <div class="stat-label">Max Used</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="search-card">
      <el-form :inline="true" :model="query">
        <el-form-item label="User ID">
          <el-input-number v-model="query.userId" :min="1" :clearable="true" style="width: 150px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="loading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="userId" label="User ID" width="90" />
        <el-table-column label="Storage Used" min-width="150">
          <template #default="{ row }">{{ formatBytes(row.totalUsed) }}</template>
        </el-table-column>
        <el-table-column prop="createdAt" label="Created" width="170">
          <template #default="{ row }">{{ row.createdAt || '-' }}</template>
        </el-table-column>
        <el-table-column prop="updatedAt" label="Updated" width="170">
          <template #default="{ row }">{{ row.updatedAt || '-' }}</template>
        </el-table-column>
        <el-table-column label="Actions" width="100" fixed="right">
          <template #default="{ row }">
            <el-button link type="danger" size="small" :icon="Delete" @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        v-model:current-page="query.pageNum"
        v-model:page-size="query.pageSize"
        :page-sizes="PAGE_SIZE_OPTIONS"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
        class="pagination"
      />
    </el-card>
  </div>
</template>

<style scoped>
.media-container { padding: 20px; }
.stat-row { margin-bottom: 20px; }
.stat-value { font-size: 24px; font-weight: bold; color: #303133; text-align: center; }
.stat-label { font-size: 13px; color: #909399; text-align: center; margin-top: 8px; }
.search-card { margin-bottom: 20px; }
.table-card { margin-bottom: 20px; }
.pagination { margin-top: 20px; justify-content: flex-end; }
</style>
