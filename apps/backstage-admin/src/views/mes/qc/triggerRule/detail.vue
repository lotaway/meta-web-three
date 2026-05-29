<template>
  <div class="trigger-rule-detail">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.triggerRule.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.qc.triggerRule.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="t('mes.qc.triggerRule.id')">
          {{ formData.id }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.ruleCode')">
          {{ formData.ruleCode }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.ruleName')">
          {{ formData.ruleName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.triggerType')">
          {{ getTriggerTypeText(formData.triggerType || '') }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.targetEntity')">
          {{ formData.targetEntity }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.priority')">
          {{ formData.priority }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.status')">
          <el-tag :type="formData.status === 'ACTIVE' ? 'success' : 'info'">
            {{ formData.status === 'ACTIVE' ? t('mes.qc.triggerRule.statusActive') : t('mes.qc.triggerRule.statusInactive') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.conditionExpression')" :span="2">
          {{ formData.conditionExpression }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.actionType')">
          {{ formData.actionType }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.actionConfig')">
          {{ formData.actionConfig }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.description')" :span="2">
          {{ formData.description }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.createdAt')">
          {{ formData.createdAt }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.triggerRule.updatedAt')">
          {{ formData.updatedAt }}
        </el-descriptions-item>
      </el-descriptions>

      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit">{{ t('common.edit') }}</el-button>
        <el-button @click="handleBack">{{ t('mes.qc.triggerRule.back') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { QcTriggerRule } from '@/apis/qc'
import { getTriggerRuleByIdAPI } from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const loading = ref(false)
const formData = reactive<QcTriggerRule>({
  id: 0,
  ruleCode: '',
  ruleName: '',
  triggerType: 'MANUAL',
  targetEntity: '',
  conditionExpression: '',
  actionType: '',
  actionConfig: '',
  priority: 0,
  status: 'INACTIVE',
  description: ''
})

const { t } = useI18n()

const getTriggerTypeText = (type: string) => {
  const textMap: Record<string, string> = {
    AUTO: t('mes.qc.triggerRule.triggerTypeAuto'),
    MANUAL: t('mes.qc.triggerRule.triggerTypeManual'),
    SCHEDULED: t('mes.qc.triggerRule.triggerTypeScheduled'),
    EVENT_BASED: t('mes.qc.triggerRule.triggerTypeEventBased')
  }
  return textMap[type] || type
}

const loadData = async () => {
  if (!route.query.id) return
  
  loading.value = true
  try {
    const res = await getTriggerRuleByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.triggerRule.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleEdit = () => {
  router.push({ path: '/mes/triggerRule/form', query: { id: route.query.id } })
}

const handleBack = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.trigger-rule-detail {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
}

.action-buttons {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}
</style>