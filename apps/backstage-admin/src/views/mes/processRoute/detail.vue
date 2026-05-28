<template>
  <div class="process-route-detail">
    <el-card v-loading="loading">
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.processRoute.detail') }}</span>
          <div class="header-actions">
            <el-button @click="handleBack">{{ t('common.cancelText') }}</el-button>
            <el-button type="primary" @click="handleEdit" v-if="routeData.status === 'DRAFT'">{{ t('common.edit') }}</el-button>
            <el-button type="warning" @click="handleActivate" v-if="routeData.status === 'DRAFT'">{{ t('mes.processRoute.activate') }}</el-button>
            <el-button type="info" @click="handleArchive" v-if="routeData.status === 'ACTIVE'">{{ t('mes.processRoute.archive') }}</el-button>
            <el-button type="success" @click="handleValidate">{{ t('mes.processRoute.validate') }}</el-button>
          </div>
        </div>
      </template>

      <el-descriptions :column="2" border v-if="routeData.id">
        <el-descriptions-item :label="t('common.id')">{{ routeData.id }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.routeCode')">{{ routeData.routeCode }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.routeName')">{{ routeData.routeName }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.productCode')">{{ routeData.productCode }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.version')">{{ routeData.version }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.status')">
          <el-tag :type="getStatusType(routeData.status)">
            {{ getStatusText(routeData.status) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.createdAt')">{{ routeData.createdAt }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.updatedAt')">{{ routeData.updatedAt }}</el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">{{ t('mes.processRoute.processFlowChart') }}</el-divider>

      <div class="steps-flow" v-if="routeData.steps && routeData.steps.length > 0">
        <el-steps :active="routeData.steps.length" align-center finish-status="success">
          <el-step 
            v-for="step in sortedSteps" 
            :key="step.stepNo"
            :title="step.processName || t('mes.processRoute.step') + step.stepNo"
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
      <el-empty v-else :description="t('mes.processRoute.noSteps')" />

      <el-divider content-position="left">{{ t('mes.processRoute.stepDetails') }}</el-divider>

      <el-table :data="sortedSteps" border stripe v-if="routeData.steps?.length">
        <el-table-column :label="t('mes.processRoute.stepNo')" prop="stepNo" width="80" align="center" />
        <el-table-column :label="t('mes.processRoute.processCode')" prop="processCode" width="120" />
        <el-table-column :label="t('mes.processRoute.processName')" prop="processName" width="150" />
        <el-table-column :label="t('mes.processRoute.workstationId')" prop="workstationId" width="100" />
        <el-table-column :label="t('mes.processRoute.standardTime')" prop="standardTime" width="140" align="right">
          <template #default="{ row }">
            {{ row.standardTime || '-' }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processRoute.qualityCheckpoint')" prop="qualityCheckpoint" width="100" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.qualityCheckpoint === 'YES'" type="success" size="small">{{ t('mes.processRoute.yes') }}</el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processRoute.predecessorStep')" width="120">
          <template #default="{ row }">
            {{ row.predecessorStepNo ? t('mes.processRoute.step') + row.predecessorStepNo : t('mes.processRoute.noPredecessor') }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processRoute.successorStep')" width="120">
          <template #default="{ row }">
            {{ row.successorStepNo ? t('mes.processRoute.step') + row.successorStepNo : t('mes.processRoute.noPredecessor') }}
          </template>
        </el-table-column>
      </el-table>

      <!-- 验证结果 -->
      <el-alert
        v-if="validationResult !== null"
        :title="validationResult ? t('mes.processRoute.validateSuccess') : t('mes.processRoute.validateFailed')"
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
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ProcessRoute, ProcessStep } from '@/apis/processRoute'
import { 
  getProcessRouteByIdAPI,
  activateProcessRouteAPI,
  archiveProcessRouteAPI,
  validateProcessRouteAPI
} from '@/apis/processRoute'

const { t } = useI18n()

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
    DRAFT: t('mes.processRoute.statusDraft'),
    ACTIVE: t('mes.processRoute.statusActive'),
    ARCHIVED: t('mes.processRoute.statusArchived')
  }
  return map[status || ''] || status
}

const getStepDescription = (step: ProcessStep) => {
  const parts: string[] = []
  if (step.processCode) parts.push(step.processCode)
  if (step.standardTime) parts.push(`${step.standardTime}${t('mes.processRoute.minutes')}`)
  if (step.qualityCheckpoint === 'YES') parts.push(t('mes.processRoute.qualityCheckpoint'))
  return parts.join(' | ') || '-'
}

const loadData = async () => {
  if (!routeId.value) {
    ElMessage.error(t('mes.processRoute.paramError'))
    router.push('/mes/process-route')
    return
  }

  loading.value = true
  try {
    const data = await getProcessRouteByIdAPI(routeId.value)
    if (data) {
      Object.assign(routeData, data)
    } else {
      ElMessage.error(t('mes.processRoute.dataNotExist'))
      router.push('/mes/process-route')
    }
  } catch (error: unknown) {
    ElMessage.error(t('mes.processRoute.loadFailed'))
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
    await ElMessageBox.confirm(t('mes.processRoute.confirmActivate'), t('common.warning'), { type: 'warning' })
    await activateProcessRouteAPI(routeId.value)
    ElMessage.success(t('mes.processRoute.activateSuccess'))
    loadData()
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.processRoute.activateFailed'))
    }
  }
}

const handleArchive = async () => {
  try {
    await ElMessageBox.confirm(t('mes.processRoute.confirmArchive'), t('common.warning'), { type: 'warning' })
    await archiveProcessRouteAPI(routeId.value)
    ElMessage.success(t('mes.processRoute.archiveSuccess'))
    loadData()
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.processRoute.archiveFailed'))
    }
  }
}

const handleValidate = async () => {
  try {
    const data = await validateProcessRouteAPI(routeId.value)
    validationResult.value = data.validationResult ?? false
    validationMessage.value = data.validationMessage || ''
    if (validationResult.value) {
      ElMessage.success(t('mes.processRoute.validateSuccess'))
    } else {
      ElMessage.warning(t('mes.processRoute.validateFailed') + ': ' + validationMessage.value)
    }
  } catch (error) {
    ElMessage.error(t('mes.processRoute.validateFailed'))
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