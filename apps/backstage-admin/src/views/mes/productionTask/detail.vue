<template>
  <div class="app-container">
    <el-page-header :title="t('mes.productionTask.detail')" @back="goBack">
      <template #content>
        <span class="text-lg">{{ t('mes.productionTask.taskNo') }}: {{ task?.taskNo }}</span>
      </template>
    </el-page-header>

    <el-card v-loading="loading" class="mt-4">
      <el-descriptions :column="2" border>
        <el-descriptions-item :label="t('mes.productionTask.taskNo')">
          {{ task?.taskNo }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.workOrderNo')">
          {{ task?.workOrderNo }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.processName')">
          {{ task?.processName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.workstationId')">
          {{ task?.workstationId }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.quantity')">
          {{ task?.quantity }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.completedQuantity')">
          {{ task?.completedQuantity }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.qualifiedQuantity')">
          {{ task?.qualifiedQuantity }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.defectiveQuantity')">
          {{ task?.defectiveQuantity }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.operatorName')">
          {{ task?.operatorName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.status')">
          <el-tag :type="getStatusType(task?.status || '')">
            {{ t(`mes.productionTask.status${task?.status}`) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.estimatedDuration')">
          {{ task?.estimatedDurationMinutes }} {{ t('mes.processRoute.minutes') }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.actualDuration')">
          {{ task?.actualDurationMinutes }} {{ t('mes.processRoute.minutes') }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.startTime')">
          {{ task?.startTime ? formatDateTime(task.startTime) : '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.productionTask.endTime')">
          {{ task?.endTime ? formatDateTime(task.endTime) : '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('common.createdAt')">
          {{ task?.createdAt ? formatDateTime(task.createdAt) : '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.processRoute.updatedAt')">
          {{ task?.updatedAt ? formatDateTime(task.updatedAt) : '-' }}
        </el-descriptions-item>
      </el-descriptions>

      <div class="mt-4">
        <el-button type="primary" :icon="Edit" @click="handleUpdate">
          {{ t('common.edit') }}
        </el-button>
        <el-button 
          v-if="task?.status === 'PENDING'" 
          type="success" 
          @click="handleStart"
        >
          {{ t('mes.productionTask.start') }}
        </el-button>
        <el-button 
          v-if="task?.status === 'IN_PROGRESS'" 
          type="warning" 
          @click="handleComplete"
        >
          {{ t('mes.productionTask.complete') }}
        </el-button>
        <el-button 
          v-if="task?.status === 'QUALITY_CHECK'" 
          type="success" 
          @click="handlePassQc"
        >
          {{ t('mes.productionTask.passQc') }}
        </el-button>
        <el-button 
          v-if="task?.status === 'QUALITY_CHECK'" 
          type="danger" 
          @click="handleFailQc"
        >
          {{ t('mes.productionTask.failQc') }}
        </el-button>
      </div>
    </el-card>

    <el-dialog v-model="completeDialogVisible" :title="t('mes.productionTask.complete')" width="400px">
      <el-form :model="completeForm" label-width="120px">
        <el-form-item :label="t('mes.productionTask.qualified')">
          <el-input-number v-model="completeForm.qualified" :min="0" :max="task?.quantity" />
        </el-form-item>
        <el-form-item :label="t('mes.productionTask.defective')">
          <el-input-number v-model="completeForm.defective" :min="0" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="completeDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="confirmComplete">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { Edit } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { 
  getTaskByIdAPI, 
  startTaskAPI, 
  completeTaskAPI,
  passQualityCheckAPI,
  failQualityCheckAPI,
  type ProductionTask 
} from '@/apis/productionTask'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const task = ref<ProductionTask>()
const loading = ref(false)
const completeDialogVisible = ref(false)
const completeForm = ref({ qualified: 0, defective: 0 })

const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    PENDING: 'info',
    IN_PROGRESS: 'warning',
    QUALITY_CHECK: 'warning',
    COMPLETED: 'success',
    CANCELLED: 'info',
    ON_HOLD: 'warning',
    SCRAPPED: 'danger',
  }
  return typeMap[status] || 'info'
}

const formatDateTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return date.toLocaleString()
}

const goBack = () => {
  router.back()
}

const loadTask = async () => {
  loading.value = true
  try {
    const id = Number(route.query.id)
    const response = await getTaskByIdAPI(id)
    task.value = response.data
  } catch (error) {
    ElMessage.error(t('mes.productionTask.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleUpdate = () => {
  router.push({ name: 'productionTaskForm', query: { id: task.value?.id } })
}

const handleStart = async () => {
  try {
    const response = await startTaskAPI(task.value!.id!)
    task.value = response.data
    ElMessage.success(t('mes.productionTask.startSuccess'))
  } catch {
    ElMessage.error(t('mes.productionTask.startFailed'))
  }
}

const handleComplete = () => {
  completeForm.value = { qualified: task.value?.quantity || 0, defective: 0 }
  completeDialogVisible.value = true
}

const confirmComplete = async () => {
  try {
    const response = await completeTaskAPI(task.value!.id!, completeForm.value)
    task.value = response.data
    completeDialogVisible.value = false
    ElMessage.success(t('mes.productionTask.completeSuccess'))
  } catch {
    ElMessage.error(t('mes.productionTask.completeFailed'))
  }
}

const handlePassQc = async () => {
  try {
    const response = await passQualityCheckAPI(task.value!.id!)
    task.value = response.data
    ElMessage.success(t('mes.productionTask.passQcSuccess'))
  } catch {
    ElMessage.error(t('mes.productionTask.passQcFailed'))
  }
}

const handleFailQc = async () => {
  try {
    const response = await failQualityCheckAPI(task.value!.id!)
    task.value = response.data
    ElMessage.success(t('mes.productionTask.failQcSuccess'))
  } catch {
    ElMessage.error(t('mes.productionTask.failQcFailed'))
  }
}

onMounted(() => {
  loadTask()
})
</script>

<style scoped>
.mt-4 {
  margin-top: 16px;
}
.text-lg {
  font-size: 18px;
}
</style>
