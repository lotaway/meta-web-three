<template>
  <div class="non-conformance-detail-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.nonConformance.detail') }}</span>
          <el-button @click="goBack">{{ t('mes.qc.nonConformance.back') }}</el-button>
        </div>
      </template>

      <div v-loading="loading">
        <el-descriptions :column="2" border v-if="detail">
          <el-descriptions-item :label="t('mes.qc.nonConformance.id')">
            {{ detail.id }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.dispositionCode')">
            {{ detail.dispositionCode }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.dispositionName')">
            {{ detail.dispositionName }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.type')">
            {{ getTypeText(detail.type) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.isEnabled')">
            <el-tag :type="detail.isEnabled ? 'success' : 'info'">
              {{ detail.isEnabled ? t('common.enable') : t('common.disable') }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.sortOrder')">
            {{ detail.sortOrder }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.createdAt')">
            {{ detail.createdAt }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.qc.nonConformance.updatedAt')">
            {{ detail.updatedAt }}
          </el-descriptions-item>
        </el-descriptions>

        <el-divider>{{ t('mes.qc.nonConformance.steps') }}</el-divider>

        <el-table :data="detail?.steps" border stripe v-if="detail?.steps?.length">
          <el-table-column :label="t('mes.qc.nonConformance.stepOrder')" prop="stepOrder" width="100" />
          <el-table-column :label="t('mes.qc.nonConformance.stepName')" prop="stepName" width="180" />
          <el-table-column :label="t('mes.qc.nonConformance.action')" prop="action" width="120">
            <template #default="{ row }">
              {{ getActionText(row.action) }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.qc.nonConformance.assigneeRole')" prop="assigneeRole" width="150" />
          <el-table-column :label="t('mes.qc.nonConformance.requiresApproval')" prop="requiresApproval" width="140">
            <template #default="{ row }">
              {{ row.requiresApproval ? t('common.yes') : t('common.no') }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.qc.nonConformance.timeoutHours')" prop="timeoutHours" width="140" />
        </el-table>

        <el-empty v-else :description="t('message.noData')" />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import type { NonConformanceDisposition } from '@/apis/qc'
import { getNonConformanceDispositionByIdAPI } from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const detail = ref<NonConformanceDisposition | null>(null)

const getTypeText = (type: string) => {
  const typeMap: Record<string, string> = {
    SCRAP: t('mes.qc.nonConformance.typeScrap'),
    REWORK: t('mes.qc.nonConformance.typeRework'),
    RETURN: t('mes.qc.nonConformance.typeReturn'),
    USE_AS_IS: t('mes.qc.nonConformance.typeUseAsIs'),
    '降级使用': t('mes.qc.nonConformance.type降级使用')
  }
  return typeMap[type] || type
}

const getActionText = (action: string) => {
  const actionMap: Record<string, string> = {
    APPROVE: t('mes.qc.nonConformance.actionApprove'),
    NOTIFY: t('mes.qc.nonConformance.actionNotify'),
    EXECUTE: t('mes.qc.nonConformance.actionExecute'),
    RECORD: t('mes.qc.nonConformance.actionRecord')
  }
  return actionMap[action] || action
}

const loadData = async () => {
  if (!route.query.id) {
    ElMessage.error(t('message.loadFailed'))
    return
  }

  loading.value = true
  try {
    const res = await getNonConformanceDispositionByIdAPI(Number(route.query.id))
    detail.value = res.data
  } catch (error) {
    ElMessage.error(t('message.loadFailed'))
  } finally {
    loading.value = false
  }
}

const goBack = () => {
  router.push({ path: '/mes/nonConformance' })
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.non-conformance-detail-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: bold;
}
</style>