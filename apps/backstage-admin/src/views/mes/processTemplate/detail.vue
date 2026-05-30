<template>
  <div class="process-template-detail-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.processTemplate.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.processTemplate.back') }}</el-button>
        </div>
      </template>

      <div v-loading="loading">
        <el-descriptions :column="2" border v-if="detail">
          <el-descriptions-item :label="t('mes.processTemplate.id')">
            {{ detail.id }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.templateCode')">
            {{ detail.templateCode }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.templateName')">
            {{ detail.templateName }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.version')">
            {{ detail.version }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.status')">
            <el-tag :type="getStatusType(detail.status)">
              {{ getStatusText(detail.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.createdAt')">
            {{ formatDateTime(detail.createdAt) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.description')" :span="2">
            {{ detail.description || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.processTemplate.flowData')" :span="2">
            <pre class="flow-data-preview">{{ detail.flowData || '-' }}</pre>
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit" v-if="detail">
          {{ t('mes.processTemplate.edit') }}
        </el-button>
        <el-button @click="handleBack">{{ t('mes.processTemplate.back') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { getProcessTemplateAPI } from '@/apis/processFlow'
import type { ProcessFlowTemplate } from '@/apis/processFlow'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const detail = ref<ProcessFlowTemplate | null>(null)

const getStatusType = (status?: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const statusMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    DRAFT: 'info',
    PUBLISHED: 'success',
    ARCHIVED: 'warning',
  }
  return statusMap[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const textMap: Record<string, string> = {
    DRAFT: t('mes.processTemplate.statusDraft'),
    PUBLISHED: t('mes.processTemplate.statusPublished'),
    ARCHIVED: t('mes.processTemplate.statusArchived'),
  }
  return textMap[status || ''] || status
}

const formatDateTime = (dateStr?: string) => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

const handleEdit = () => {
  router.push(`/mes/process-template/form?id=${route.query.id}`)
}

const handleBack = () => {
  router.back()
}

const loadData = async () => {
  if (!route.query.id) {
    ElMessage.error(t('mes.processTemplate.paramError'))
    router.push('/mes/process-template')
    return
  }

  loading.value = true
  try {
    const res = await getProcessTemplateAPI(Number(route.query.id))
    detail.value = res.data
  } catch (error) {
    ElMessage.error(t('mes.processTemplate.loadFailed'))
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.process-template-detail-container {
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