<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, Refresh, VideoPlay, VideoPause } from '@element-plus/icons-vue'
import {
  getRecommendationRuleListAPI,
  getRecommendationStatisticsAPI,
  createRecommendationRuleAPI,
  updateRecommendationRuleAPI,
  deleteRecommendationRuleAPI,
  activateRecommendationRuleAPI,
  pauseRecommendationRuleAPI,
  type RecommendationRule,
  type RecommendationRuleQueryParams,
  type CreateRecommendationRuleParams
} from '@/apis/recommendation'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

// State
const loading = ref(false)
const statistics = ref({
  totalRules: 0,
  activeRules: 0,
  totalRecommendations: 0,
  totalClicks: 0,
  totalConversions: 0,
  avgClickThroughRate: 0,
  avgConversionRate: 0,
  sceneDistribution: {} as Record<string, number>,
  algorithmDistribution: {} as Record<string, number>
})
const activeTab = ref('rules')

// Query params
const query = reactive<RecommendationRuleQueryParams>({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  ruleName: '',
  scene: '',
  status: ''
})

const ruleList = ref<RecommendationRule[]>([])
const ruleTotal = ref(0)

// Dialog state
const dialogVisible = ref(false)
const dialogTitle = ref('')
const ruleForm = reactive<CreateRecommendationRuleParams>({
  ruleName: '',
  scene: '',
  type: 'BOOST',
  priority: 0,
  maxItems: 10,
  conditions: '',
  exclusions: '',
  boostFactor: 1.0
})
const ruleFormRef = ref()
const editingId = ref<number | undefined>()

const sceneOptions = [
  { label: 'Home', value: 'HOME' },
  { label: 'Product Detail', value: 'PRODUCT_DETAIL' },
  { label: 'Cart', value: 'CART' },
  { label: 'Search', value: 'SEARCH' },
  { label: 'Category', value: 'CATEGORY' },
  { label: 'Order Complete', value: 'ORDER_COMPLETE' }
]

const ruleTypeOptions = [
  { label: 'BOOST', value: 'BOOST' },
  { label: 'FILTER', value: 'FILTER' },
  { label: 'RE_RANK', value: 'RE_RANK' },
  { label: 'EXCLUDE', value: 'EXCLUDE' },
  { label: 'COLLABORATIVE', value: 'COLLABORATIVE' },
  { label: 'CONTENT_BASED', value: 'CONTENT_BASED' },
  { label: 'HYBRID', value: 'HYBRID' },
  { label: 'POPULARITY', value: 'POPULARITY' }
]

const statusOptions = [
  { label: 'DRAFT', value: 'DRAFT' },
  { label: 'ACTIVE', value: 'ACTIVE' },
  { label: 'PAUSED', value: 'PAUSED' },
  { label: 'ARCHIVED', value: 'ARCHIVED' }
]

const getStatusTagType = (status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    PAUSED: 'warning',
    ARCHIVED: 'danger'
  }
  return map[status] || 'info'
}

const getTypeTagType = (type: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    BOOST: 'success',
    FILTER: 'warning',
    RE_RANK: 'primary',
    EXCLUDE: 'danger',
    COLLABORATIVE: 'primary',
    CONTENT_BASED: 'success',
    HYBRID: 'warning',
    POPULARITY: 'info'
  }
  return map[type] || 'info'
}

// Methods
const getStatistics = async () => {
  try {
    const res = await getRecommendationStatisticsAPI()
    statistics.value = res.data
  } catch {
    ElMessage.error(t('common.queryFailed'))
  }
}

const getRuleList = async () => {
  loading.value = true
  try {
    const res = await getRecommendationRuleListAPI(query)
    ruleList.value = res.data.list
    ruleTotal.value = res.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  query.pageNum = 1
  getRuleList()
}

const handleReset = () => {
  Object.assign(query, {
    pageNum: 1,
    pageSize: DEFAULT_PAGE_SIZE,
    ruleName: '',
    scene: '',
    status: ''
  })
  getRuleList()
}

const handleAdd = () => {
  dialogTitle.value = t('common.add')
  editingId.value = undefined
  Object.assign(ruleForm, {
    ruleName: '',
    scene: '',
    type: 'BOOST',
    priority: 0,
    maxItems: 10,
    conditions: '',
    exclusions: '',
    boostFactor: '1.0'
  })
  dialogVisible.value = true
}

const handleEdit = (row: RecommendationRule) => {
  dialogTitle.value = t('common.edit')
  editingId.value = row.id
  Object.assign(ruleForm, {
    ruleName: row.ruleName,
    scene: row.scene,
    type: row.type as string,
    priority: row.priority,
    maxItems: row.maxItems,
    conditions: row.conditions || '',
    exclusions: row.exclusions || '',
    boostFactor: row.boostFactor ? Number(row.boostFactor) : 1.0
  })
  dialogVisible.value = true
}

const handleDelete = async (row: RecommendationRule) => {
  try {
    await ElMessageBox.confirm('Delete this rule?', t('common.warning'), {
      type: 'warning'
    })
    await deleteRecommendationRuleAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    getRuleList()
    getStatistics()
  } catch {
    // User cancelled or error
  }
}

const handleActivate = async (row: RecommendationRule) => {
  try {
    await activateRecommendationRuleAPI(row.id!)
    ElMessage.success(t('common.operationSuccess'))
    getRuleList()
    getStatistics()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handlePause = async (row: RecommendationRule) => {
  try {
    await pauseRecommendationRuleAPI(row.id!)
    ElMessage.success(t('common.operationSuccess'))
    getRuleList()
    getStatistics()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handleSubmit = async () => {
  try {
    if (editingId.value) {
      await updateRecommendationRuleAPI(editingId.value, ruleForm)
      ElMessage.success(t('common.updateSuccess'))
    } else {
      await createRecommendationRuleAPI(ruleForm)
      ElMessage.success(t('common.createSuccess'))
    }
    dialogVisible.value = false
    getRuleList()
    getStatistics()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

onMounted(() => {
  getStatistics()
  getRuleList()
})
</script>

<template>
  <div class="recommendation-container">
    <!-- Statistics Dashboard -->
    <el-card class="dashboard-card">
      <div class="dashboard-header">
        <span class="dashboard-title">Recommendation Dashboard</span>
      </div>
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="metric-box">
            <div class="metric-value">{{ statistics.totalRules }}</div>
            <div class="metric-label">Total Rules</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-box active">
            <div class="metric-value">{{ statistics.activeRules }}</div>
            <div class="metric-label">Active Rules</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-box">
            <div class="metric-value">{{ statistics.totalRecommendations }}</div>
            <div class="metric-label">Total Recommendations</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-box">
            <div class="metric-value">{{ statistics.totalClicks }}</div>
            <div class="metric-label">Total Clicks</div>
          </div>
        </el-col>
      </el-row>
      <el-row :gutter="20" class="second-row">
        <el-col :span="6">
          <div class="metric-box">
            <div class="metric-value">{{ statistics.totalConversions }}</div>
            <div class="metric-label">Total Conversions</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-box highlight">
            <div class="metric-value">{{ statistics.avgClickThroughRate }}%</div>
            <div class="metric-label">Avg Click-Through Rate</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-box highlight">
            <div class="metric-value">{{ statistics.avgConversionRate }}%</div>
            <div class="metric-label">Avg Conversion Rate</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-box">
            <div class="metric-value">
              <el-tag v-if="statistics.avgClickThroughRate > 3.5" type="success">Good</el-tag>
              <el-tag v-else-if="statistics.avgClickThroughRate > 1.5" type="warning">Normal</el-tag>
              <el-tag v-else type="danger">Low</el-tag>
            </div>
            <div class="metric-label">CTR Health</div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- Scene & Algorithm Distribution -->
    <el-row :gutter="20" class="distribution-row">
      <el-col :span="12">
        <el-card class="distribution-card">
          <template #header>
            <span>Scene Distribution</span>
          </template>
          <div class="distribution-list">
            <div v-for="(count, scene) in statistics.sceneDistribution" :key="scene" class="dist-item">
              <span class="dist-label">{{ scene }}</span>
              <el-progress :percentage="Math.round(count / Math.max(...Object.values(statistics.sceneDistribution)) * 100)" :stroke-width="16" />
              <span class="dist-count">{{ count }}</span>
            </div>
            <div v-if="Object.keys(statistics.sceneDistribution).length === 0" class="empty-hint">No data</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card class="distribution-card">
          <template #header>
            <span>Algorithm Distribution</span>
          </template>
          <div class="distribution-list">
            <div v-for="(count, algo) in statistics.algorithmDistribution" :key="algo" class="dist-item">
              <span class="dist-label">{{ algo }}</span>
              <el-progress :percentage="Math.round(count / Math.max(...Object.values(statistics.algorithmDistribution)) * 100)" :stroke-width="16" />
              <span class="dist-count">{{ count }}</span>
            </div>
            <div v-if="Object.keys(statistics.algorithmDistribution).length === 0" class="empty-hint">No data</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Rules and Recommendation Records Tabs -->
    <el-card class="section-card">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="Rule Management" name="rules">
          <!-- Search Form -->
          <el-form :inline="true" :model="query" class="search-form">
            <el-form-item label="Rule Name">
              <el-input v-model="query.ruleName" placeholder="Rule name" clearable />
            </el-form-item>
            <el-form-item label="Scene">
              <el-select v-model="query.scene" placeholder="Select scene" clearable>
                <el-option v-for="item in sceneOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item label="Status">
              <el-select v-model="query.status" placeholder="Select status" clearable>
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
              <el-button :icon="Refresh" @click="handleReset">{{ t('common.reset') }}</el-button>
              <el-button type="success" :icon="Plus" @click="handleAdd">Add Rule</el-button>
            </el-form-item>
          </el-form>

          <!-- Rule List Table -->
          <el-table :data="ruleList" v-loading="loading" stripe>
            <el-table-column prop="id" label="ID" width="60" />
            <el-table-column prop="ruleName" label="Rule Name" width="150" />
            <el-table-column prop="scene" label="Scene" width="120" />
            <el-table-column prop="type" label="Type" width="120">
              <template #default="{ row }">
                <el-tag :type="getTypeTagType(row.type as string)">
                  {{ row.type }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="Status" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusTagType(row.status as string)">
                  {{ row.status }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="priority" label="Priority" width="70" />
            <el-table-column prop="maxItems" label="Max Items" width="80" />
            <el-table-column prop="boostFactor" label="Boost Factor" width="90" />
            <el-table-column prop="conditions" label="Conditions" min-width="140" show-overflow-tooltip />
            <el-table-column label="Actions" width="180" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" size="small" @click="handleEdit(row)">Edit</el-button>
                <el-button v-if="row.status === 'DRAFT' || row.status === 'PAUSED'" link type="success" size="small" @click="handleActivate(row)">
                  <el-icon><VideoPlay /></el-icon>
                </el-button>
                <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="handlePause(row)">
                  <el-icon><VideoPause /></el-icon>
                </el-button>
                <el-button link type="danger" size="small" @click="handleDelete(row)">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-pagination
            v-model:current-page="query.pageNum"
            v-model:page-size="query.pageSize"
            :page-sizes="PAGE_SIZE_OPTIONS"
            :total="ruleTotal"
            layout="total, sizes, prev, pager, next, jumper"
            @size-change="getRuleList"
            @current-change="getRuleList"
            class="pagination"
          />
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- Rule Form Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px">
      <el-form ref="ruleFormRef" :model="ruleForm" label-width="120px">
        <el-form-item label="Rule Name" required>
          <el-input v-model="ruleForm.ruleName" placeholder="Enter rule name" />
        </el-form-item>
        <el-form-item label="Scene" required>
          <el-select v-model="ruleForm.scene" placeholder="Select scene">
            <el-option v-for="item in sceneOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Type" required>
          <el-select v-model="ruleForm.type" placeholder="Select type">
            <el-option v-for="item in ruleTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Priority">
          <el-input-number v-model="ruleForm.priority" :min="0" :max="100" />
        </el-form-item>
        <el-form-item label="Max Items">
          <el-input-number v-model="ruleForm.maxItems" :min="1" :max="100" />
        </el-form-item>
        <el-form-item label="Boost Factor">
          <el-input-number v-model="ruleForm.boostFactor" :min="0.1" :max="10" :step="0.1" :precision="1" />
        </el-form-item>
        <el-form-item label="Conditions">
          <el-input v-model="ruleForm.conditions" type="textarea" :rows="3" placeholder="Enter conditions (JSON or SQL like)" />
        </el-form-item>
        <el-form-item label="Exclusions">
          <el-input v-model="ruleForm.exclusions" type="textarea" :rows="3" placeholder="Enter exclusions (comma separated SKUs)" />
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
.recommendation-container {
  padding: 20px;
}

.dashboard-card {
  margin-bottom: 20px;
}

.dashboard-header {
  margin-bottom: 16px;
}

.dashboard-title {
  font-size: 18px;
  font-weight: 600;
}

.metric-box {
  text-align: center;
  padding: 12px 0;
  border-right: 1px solid #ebeef5;
}

.metric-box.active {
  border-top: 3px solid #67c23a;
  padding-top: 9px;
}

.metric-box.highlight {
  border-top: 3px solid #409eff;
  padding-top: 9px;
}

.metric-value {
  font-size: 26px;
  font-weight: bold;
  color: #303133;
}

.metric-label {
  font-size: 13px;
  color: #909399;
  margin-top: 5px;
}

.second-row {
  margin-top: 16px;
}

.distribution-row {
  margin-bottom: 20px;
}

.distribution-card {
  margin-bottom: 20px;
}

.distribution-list {
  padding: 0 4px;
}

.dist-item {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
  gap: 12px;
}

.dist-label {
  width: 140px;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
}

.dist-count {
  width: 40px;
  text-align: right;
  font-size: 13px;
  color: #606266;
}

.empty-hint {
  text-align: center;
  color: #c0c4cc;
  padding: 20px;
  font-size: 14px;
}

.section-card {
  margin-bottom: 20px;
}

.search-form {
  margin-bottom: 16px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>