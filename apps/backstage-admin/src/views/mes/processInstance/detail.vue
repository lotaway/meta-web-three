<template>
  <div class="process-instance-detail-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.processInstance.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.processInstance.back') }}</el-button>
        </div>
      </template>

      <div v-loading="loading">
        <el-descriptions :column="2" border v-if="detail">
          <el-descriptions-item :label="t('mes.processInstance.id')">
            {{ detail.id }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.templateName')">
            {{ detail.templateName }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.businessType')">
            {{ getBusinessTypeText(detail.businessType) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.businessKey')">
            {{ detail.businessKey }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.status')">
            <el-tag :type="getStatusType(detail.status)">
              {{ getStatusText(detail.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.currentNode')">
            {{ detail.currentNodeId || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.startedAt')">
            {{ formatDateTime(detail.startedAt) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.completedAt')">
            {{ formatDateTime(detail.completedAt) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processInstance.flowData')" :span="2">
            <pre class="flow-data-preview">{{ detail.flowData || '-' }}</pre>
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <div class="action-buttons">
        <el-button type="primary" @click="handleComplete" v-if="detail?.status === 'RUNNING'">
          {{ t('mes.processInstance.complete') }}
        </el-button>
        <el-button type="danger" @click="handleTerminate" v-if="detail?.status === 'RUNNING'">
          {{ t('mes.processInstance.terminate') }}
        </el-button>
        <el-button @click="handleBack">{{ t('mes.processInstance.back') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  getProcessInstanceAPI,
  completeProcessInstanceAPI,
  terminateProcessInstanceAPI,
} from '@/apis/processFlow'
import type { ProcessFlowInstance } from '@/apis/processFlow'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const detail = ref<ProcessFlowInstance | null>(null)

const getStatusType = (status?: string) => {
  const statusMap: Record<string, string> = {
    RUNNING: 'success',
    COMPLETED: 'info',
    TERMINATED: 'danger',
  }
  return statusMap[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const textMap: Record<string, string> = {
    RUNNING: t('mes.processInstance.statusRunning'),
    COMPLETED: t('mes.processInstance.statusCompleted'),
    TERMINATED: t('mes.processInstance.statusTerminated'),
  }
  return textMap[status || ''] || status
}

const getBusinessTypeText = (type?: string) => {
  const textMap: Record<string, string> = {
    WORK_ORDER: t('mes.processInstance.typeWorkOrder'),
    PRODUCTION_TASK: t('mes.processInstance.typeProductionTask'),
    QC_INSPECTION: t('mes.processInstance.typeQcInspection'),
  }
  return textMap[type || ''] || type
}

const formatDateTime = (dateStr?: string) => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

const handleBack = () => {
  router.back()
}

const handleComplete = async () => {
  try {
    await ElMessageBox.confirm(
      t('mes.processInstance.confirmComplete'),
      t('mes.processInstance.warning'),
      { confirmButtonText: t('mes.processInstance.confirm'), cancelButtonText: t('mes.processInstance.cancel'), type: 'warning' }
    )
    await completeProcessInstanceAPI(Number(route.query.id), 1)
    ElMessage.success(t('mes.processInstance.completeSuccess'))
    loadData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processInstance.completeFailed'))
    }
  }
}

const handleTerminate = async () => {
  try {
    await ElMessageBox.confirm(
      t('mes.processInstance.confirmTerminate'),
      t('mes.processInstance.warning'),
      { confirmButtonText: t('mes.processInstance.confirm'), cancelButtonText: t('mes.processInstance.cancel'), type: 'warning' }
    )
    await terminateProcessInstanceAPI(Number(route.query.id), 1)
    ElMessage.success(t('mes.processInstance.terminateSuccess'))
    loadData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processInstance.terminateFailed'))
    }
  }
}

const loadData = async () => {
  if (!route.query.id) {
    ElMessage.error(t('mes.processInstance.paramError'))
    router.push('/mes/process-instance')
    return
  }

  loading.value = true
  try {
    const res = await getProcessInstanceAPI(Number(route.query.id))
    detail.value = res.data
  } catch (error) {
    ElMessage.error(t('mes.processInstance.loadFailed'))
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.process-instance-detail-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.flow-data-preview {
  background: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 300px;
  overflow-y: auto;
}

.action-buttons {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}
</style>