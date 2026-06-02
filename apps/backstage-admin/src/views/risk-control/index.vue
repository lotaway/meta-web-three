<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, Refresh, Warning } from '@element-plus/icons-vue'
import {
  getRiskEventListAPI,
  getRiskStatisticsAPI,
  updateRiskEventStatusAPI,
  type RiskEvent,
  type RiskEventQueryParams
} from '@/apis/riskControl'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

// State
const activeTab = ref('event')
const loading = ref(false)
const statistics = ref<{
  total: number
  high: number
  medium: number
  low: number
  reviewPending: number
}>({
  total: 0,
  high: 0,
  medium: 0,
  low: 0,
  reviewPending: 0
})

// Event query params
const eventQuery = reactive<RiskEventQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  userId: '',
  scene: '',
  riskLevel: '',
  decision: '',
  status: undefined
})

const eventList = ref<RiskEvent[]>([])
const eventTotal = ref(0)

// Dialog state
const eventDialogVisible = ref(false)
const eventDetail = ref<RiskEvent>({})

const riskLevelOptions = [
  { label: 'HIGH', value: 'HIGH' },
  { label: 'MEDIUM', value: 'MEDIUM' },
  { label: 'LOW', value: 'LOW' }
]

const decisionOptions = [
  { label: 'PASS', value: 'PASS' },
  { label: 'REVIEW', value: 'REVIEW' },
  { label: 'REJECT', value: 'REJECT' }
]

const sceneOptions = [
  { label: 'ORDER', value: 'ORDER' },
  { label: 'PAYMENT', value: 'PAYMENT' },
  { label: 'LOGIN', value: 'LOGIN' },
  { label: 'REGISTER', value: 'REGISTER' }
]

const statusOptions = [
  { label: 'Pending', value: 0 },
  { label: 'Processed', value: 1 },
  { label: 'Ignored', value: 2 }
]

// Methods
const getStatistics = async () => {
  try {
    const res = await getRiskStatisticsAPI()
    statistics.value = res.data
  } catch {
    ElMessage.error(t('common.queryFailed'))
  }
}

const getEventList = async () => {
  loading.value = true
  try {
    const res = await getRiskEventListAPI(eventQuery)
    eventList.value = res.data.list
    eventTotal.value = res.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    loading.value = false
  }
}

const handleEventSearch = () => {
  eventQuery.pageNum = 1
  getEventList()
}

const handleEventReset = () => {
  Object.assign(eventQuery, {
    pageNum: 1,
    pageSize: DEFAULT_PAGE_SIZE,
    userId: '',
    scene: '',
    riskLevel: '',
    decision: '',
    status: undefined
  })
  getEventList()
}

const handleEventView = (row: RiskEvent) => {
  eventDetail.value = { ...row }
  eventDialogVisible.value = true
}

const handleEventProcess = async (row: RiskEvent) => {
  try {
    await ElMessageBox.confirm('Mark this event as processed?', t('common.warning'), {
      type: 'info'
    })
    await updateRiskEventStatusAPI(row.id!, 1)
    ElMessage.success(t('common.operationSuccess'))
    getEventList()
    getStatistics()
  } catch {
    // User cancelled or error
  }
}

const handleEventIgnore = async (row: RiskEvent) => {
  try {
    await ElMessageBox.confirm('Ignore this event?', t('common.warning'), {
      type: 'info'
    })
    await updateRiskEventStatusAPI(row.id!, 2)
    ElMessage.success(t('common.operationSuccess'))
    getEventList()
    getStatistics()
  } catch {
    // User cancelled or error
  }
}

const getRiskLevelTagType = (level: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    HIGH: 'danger',
    MEDIUM: 'warning',
    LOW: 'success'
  }
  return map[level] || 'info'
}

const getDecisionTagType = (decision: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    PASS: 'success',
    REVIEW: 'warning',
    REJECT: 'danger'
  }
  return map[decision] || 'info'
}

const formatTime = (timestamp: number) => {
  if (!timestamp) return '-'
  return new Date(timestamp).toLocaleString()
}

onMounted(() => {
  getStatistics()
  getEventList()
})
</script>

<template>
  <div class="risk-control-container">
    <el-tabs v-model="activeTab">
      <el-tab-pane label="Risk Events" name="event">
        <!-- Statistics Cards -->
        <el-row :gutter="20" class="statistics-row">
          <el-col :span="6">
            <el-card class="stat-card">
              <div class="stat-content">
                <div class="stat-value">{{ statistics.total }}</div>
                <div class="stat-label">Total Events</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat-card high">
              <div class="stat-content">
                <el-icon><Warning /></el-icon>
                <div class="stat-value">{{ statistics.high }}</div>
                <div class="stat-label">High Risk</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat-card medium">
              <div class="stat-content">
                <div class="stat-value">{{ statistics.medium }}</div>
                <div class="stat-label">Medium Risk</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat-card review">
              <div class="stat-content">
                <div class="stat-value">{{ statistics.reviewPending }}</div>
                <div class="stat-label">Pending Review</div>
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- Search Form -->
        <el-card class="search-card">
          <el-form :inline="true" :model="eventQuery">
            <el-form-item label="User ID">
              <el-input v-model="eventQuery.userId" placeholder="User ID" clearable />
            </el-form-item>
            <el-form-item label="Scene">
              <el-select v-model="eventQuery.scene" placeholder="Select scene" clearable>
                <el-option v-for="item in sceneOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item label="Risk Level">
              <el-select v-model="eventQuery.riskLevel" placeholder="Select level" clearable>
                <el-option v-for="item in riskLevelOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item label="Decision">
              <el-select v-model="eventQuery.decision" placeholder="Select decision" clearable>
                <el-option v-for="item in decisionOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleEventSearch">{{ t('common.search') }}</el-button>
              <el-button :icon="Refresh" @click="handleEventReset">{{ t('common.reset') }}</el-button>
            </el-form-item>
          </el-form>
        </el-card>

        <!-- Event List Table -->
        <el-card class="table-card">
          <el-table :data="eventList" v-loading="loading" stripe>
            <el-table-column prop="eventId" label="Event ID" width="180" />
            <el-table-column prop="userId" label="User ID" width="120" />
            <el-table-column prop="scene" label="Scene" width="100" />
            <el-table-column prop="eventType" label="Type" width="120" />
            <el-table-column prop="riskScore" label="Score" width="80" />
            <el-table-column prop="riskLevel" label="Risk Level" width="100">
              <template #default="{ row }">
                <el-tag :type="getRiskLevelTagType(row.riskLevel)">
                  {{ row.riskLevel }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="decision" label="Decision" width="100">
              <template #default="{ row }">
                <el-tag :type="getDecisionTagType(row.decision)">
                  {{ row.decision }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="row.status === 1 ? 'success' : row.status === 2 ? 'info' : 'warning'">
                  {{ row.status === 1 ? 'Processed' : row.status === 2 ? 'Ignored' : 'Pending' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="createTime" label="Create Time" width="160">
              <template #default="{ row }">
                {{ formatTime(row.createTime) }}
              </template>
            </el-table-column>
            <el-table-column label="Actions" width="180" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" size="small" @click="handleEventView(row)">View</el-button>
                <el-button v-if="row.status === 0" link type="success" size="small" @click="handleEventProcess(row)">Process</el-button>
                <el-button v-if="row.status === 0" link type="info" size="small" @click="handleEventIgnore(row)">Ignore</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-pagination
            v-model:current-page="eventQuery.pageNum"
            v-model:page-size="eventQuery.pageSize"
            :page-sizes="PAGE_SIZE_OPTIONS"
            :total="eventTotal"
            layout="total, sizes, prev, pager, next, jumper"
            @size-change="getEventList"
            @current-change="getEventList"
            class="pagination"
          />
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <!-- Event Detail Dialog -->
    <el-dialog v-model="eventDialogVisible" title="Risk Event Details" width="600px">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="Event ID">{{ eventDetail.eventId }}</el-descriptions-item>
        <el-descriptions-item label="User ID">{{ eventDetail.userId }}</el-descriptions-item>
        <el-descriptions-item label="Scene">{{ eventDetail.scene }}</el-descriptions-item>
        <el-descriptions-item label="Event Type">{{ eventDetail.eventType }}</el-descriptions-item>
        <el-descriptions-item label="Risk Score">{{ eventDetail.riskScore }}</el-descriptions-item>
        <el-descriptions-item label="Risk Level">
          <el-tag :type="getRiskLevelTagType(eventDetail.riskLevel!)">
            {{ eventDetail.riskLevel }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Decision">
          <el-tag :type="getDecisionTagType(eventDetail.decision!)">
            {{ eventDetail.decision }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Description" :span="2">{{ eventDetail.description }}</el-descriptions-item>
        <el-descriptions-item label="Details" :span="2">{{ eventDetail.details }}</el-descriptions-item>
        <el-descriptions-item label="Create Time">{{ formatTime(eventDetail.createTime!) }}</el-descriptions-item>
        <el-descriptions-item label="Update Time">{{ formatTime(eventDetail.updateTime!) }}</el-descriptions-item>
      </el-descriptions>
    </el-dialog>
  </div>
</template>

<style scoped>
.risk-control-container {
  padding: 20px;
}

.statistics-row {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
}

.stat-card.high {
  border-top: 3px solid #f56c6c;
}

.stat-card.medium {
  border-top: 3px solid #e6a23c;
}

.stat-card.review {
  border-top: 3px solid #409eff;
}

.stat-content {
  padding: 10px;
}

.stat-content .el-icon {
  font-size: 24px;
  color: #f56c6c;
  margin-bottom: 8px;
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

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>