<template>
  <div class="process-route-detail">
    <el-card v-loading="loading">
      <template #header>
        <div class="card-header">
          <span>工艺路线详情</span>
          <div class="header-actions">
            <el-button @click="handleBack">返回</el-button>
            <el-button type="primary" @click="handleEdit" v-if="routeData.status === 'DRAFT'">编辑</el-button>
            <el-button type="warning" @click="handleActivate" v-if="routeData.status === 'DRAFT'">激活</el-button>
            <el-button type="info" @click="handleArchive" v-if="routeData.status === 'ACTIVE'">归档</el-button>
            <el-button type="success" @click="handleValidate">验证</el-button>
          </div>
        </div>
      </template>

      <el-descriptions :column="2" border v-if="routeData.id">
        <el-descriptions-item label="ID">{{ routeData.id }}</el-descriptions-item>
        <el-descriptions-item label="路线编码">{{ routeData.routeCode }}</el-descriptions-item>
        <el-descriptions-item label="路线名称">{{ routeData.routeName }}</el-descriptions-item>
        <el-descriptions-item label="产品编码">{{ routeData.productCode }}</el-descriptions-item>
        <el-descriptions-item label="版本">{{ routeData.version }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="getStatusType(routeData.status)">
            {{ getStatusText(routeData.status) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ routeData.createdAt }}</el-descriptions-item>
        <el-descriptions-item label="更新时间">{{ routeData.updatedAt }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">工序流程图</el-divider>

      <div class="steps-flow" v-if="routeData.steps && routeData.steps.length > 0">
        <el-steps :active="routeData.steps.length" align-center finish-status="success">
          <el-step 
            v-for="step in sortedSteps" 
            :key="step.stepNo"
            :title="step.processName || `工序${step.stepNo}`"
            :description="getStepDescription(step)"
          >
            <template #icon>
              <div class="step-icon">
                {{ step.stepNo }}
              </div>
            </template>
          </el-step>
        </el-steps>
      </div>
      <el-empty v-else description="暂无工序步骤" />

      <el-divider content-position="left">工序详情</el-divider>

      <el-table :data="sortedSteps" border stripe v-if="routeData.steps?.length">
        <el-table-column label="序号" prop="stepNo" width="80" align="center" />
        <el-table-column label="工序编码" prop="processCode" width="120" />
        <el-table-column label="工序名称" prop="processName" width="150" />
        <el-table-column label="工位ID" prop="workstationId" width="100" />
        <el-table-column label="标准工时(分钟)" prop="standardTime" width="140" align="right">
          <template #default="{ row }">
            {{ row.standardTime || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="质检工序" prop="qualityCheckpoint" width="100" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.qualityCheckpoint === 'YES'" type="success" size="small">是</el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="前驱工序" width="120">
          <template #default="{ row }">
            {{ row.predecessorStepNo ? `步骤${row.predecessorStepNo}` : '无' }}
          </template>
        </el-table-column>
        <el-table-column label="后继工序" width="120">
          <template #default="{ row }">
            {{ row.successorStepNo ? `步骤${row.successorStepNo}` : '无' }}
          </template>
        </el-table-column>
      </el-table>

      <!-- 验证结果 -->
      <el-alert
        v-if="validationResult !== null"
        :title="validationResult ? '验证通过' : '验证失败'"
        :type="validationResult ? 'success' : 'error'"
        :description="validationMessage"
        show-icon
        style="margin-top: 20px"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ProcessRoute, ProcessStep } from '@/apis/processRoute'
import { 
  getProcessRouteByIdAPI,
  activateProcessRouteAPI,
  archiveProcessRouteAPI,
  validateProcessRouteAPI
} from '@/apis/processRoute'

const route = useRoute()
const router = useRouter()
const loading = ref(false)

const routeId = computed(() => Number(route.query.id) || 0)

const routeData = reactive<ProcessRoute>({
  id: undefined,
  routeCode: '',
  routeName: '',
  productCode: '',
  version: undefined,
  status: undefined,
  steps: [],
  createdAt: undefined,
  updatedAt: undefined,
  validationMessage: undefined,
  validationResult: undefined
})

const validationResult = ref<boolean | null>(null)
const validationMessage = ref('')

const sortedSteps = computed(() => {
  if (!routeData.steps) return []
  return [...routeData.steps].sort((a, b) => (a.stepNo || 0) - (b.stepNo || 0))
})

const getStatusType = (status?: string) => {
  const map: Record<string, string> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    ARCHIVED: ''
  }
  return map[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const map: Record<string, string> = {
    DRAFT: '草稿',
    ACTIVE: '已激活',
    ARCHIVED: '已归档'
  }
  return map[status || ''] || status
}

const getStepDescription = (step: ProcessStep) => {
  const parts: string[] = []
  if (step.processCode) parts.push(step.processCode)
  if (step.standardTime) parts.push(`${step.standardTime}分钟`)
  if (step.qualityCheckpoint === 'YES') parts.push('质检点')
  return parts.join(' | ') || '-'
}

const loadData = async () => {
  if (!routeId.value) {
    ElMessage.error('参数错误')
    router.push('/mes/process-route')
    return
  }

  loading.value = true
  try {
    const data = await getProcessRouteByIdAPI(routeId.value)
    if (data) {
      Object.assign(routeData, data)
    } else {
      ElMessage.error('数据不存在')
      router.push('/mes/process-route')
    }
  } catch (error) {
    console.error('加载失败:', error)
    ElMessage.error('加载数据失败')
  } finally {
    loading.value = false
  }
}

const handleBack = () => {
  router.push('/mes/process-route')
}

const handleEdit = () => {
  router.push({ path: '/mes/process-route/form', query: { id: routeId.value } })
}

const handleActivate = async () => {
  try {
    await ElMessageBox.confirm('确定要激活该工艺路线吗？', '提示', { type: 'warning' })
    await activateProcessRouteAPI(routeId.value)
    ElMessage.success('激活成功')
    loadData()
  } catch (error) {
    // 用户取消
  }
}

const handleArchive = async () => {
  try {
    await ElMessageBox.confirm('确定要归档该工艺路线吗？', '提示', { type: 'warning' })
    await archiveProcessRouteAPI(routeId.value)
    ElMessage.success('归档成功')
    loadData()
  } catch (error) {
    // 用户取消
  }
}

const handleValidate = async () => {
  try {
    const data = await validateProcessRouteAPI(routeId.value)
    validationResult.value = data.validationResult ?? false
    validationMessage.value = data.validationMessage || ''
    if (validationResult.value) {
      ElMessage.success('验证通过')
    } else {
      ElMessage.warning('验证失败: ' + validationMessage.value)
    }
  } catch (error) {
    ElMessage.error('验证失败')
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.process-route-detail {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.steps-flow {
  padding: 30px 20px;
  background: #fafafa;
  border-radius: 4px;
  margin: 20px 0;
}

.step-icon {
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  background: #409eff;
  color: #fff;
  border-radius: 50%;
  font-weight: bold;
}
</style>