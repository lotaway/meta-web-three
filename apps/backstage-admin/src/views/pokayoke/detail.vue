<template>
  <div class="pokayoke-rule-detail-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.pokayoke.detailTitle') }}</span>
          <div>
            <el-button @click="handleBack">{{ t('mes.pokayoke.back') }}</el-button>
            <el-button type="primary" @click="handleEdit">{{ t('mes.pokayoke.edit') }}</el-button>
          </div>
        </div>
      </template>

      <div v-loading="ruleStore.loading">
        <el-descriptions :column="2" border v-if="ruleStore.currentRule">
          <el-descriptions-item :label="t('mes.pokayoke.id')">
            {{ ruleStore.currentRule.id }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.ruleCode')">
            {{ ruleStore.currentRule.ruleCode }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.ruleName')">
            {{ ruleStore.currentRule.ruleName }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.ruleType')">
            <el-tag :type="getRuleTypeTag(ruleStore.currentRule.ruleType)">
              {{ getRuleTypeLabel(ruleStore.currentRule.ruleType) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.status')">
            <el-tag :type="getStatusTag(ruleStore.currentRule.status)">
              {{ getStatusLabel(ruleStore.currentRule.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.priority')">
            {{ ruleStore.currentRule.priority }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.workstation')">
            {{ ruleStore.currentRule.workstationId || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.errorMessage')">
            {{ ruleStore.currentRule.errorMessage || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.conditionExpression')" :span="2">
            <pre class="code-block">{{ ruleStore.currentRule.conditionExpression || '-' }}</pre>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.actionType')">
            {{ getActionTypeLabel(ruleStore.currentRule.actionType) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.actionConfig')">
            <pre class="code-block">{{ ruleStore.currentRule.actionConfig || '-' }}</pre>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.createdAt')">
            {{ formatDateTime(ruleStore.currentRule.createdAt) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.pokayoke.updatedAt')">
            {{ formatDateTime(ruleStore.currentRule.updatedAt) }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { usePokayokeRuleStore } from '@/stores/pokayokeRule'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const ruleStore = usePokayokeRuleStore()

onMounted(async () => {
  if (route.query.id) {
    await ruleStore.fetchRuleById(Number(route.query.id))
  }
})

function handleEdit() {
  router.push(`/mes/pokayoke/form?id=${route.query.id}`)
}

function handleBack() {
  router.push('/mes/pokayoke')
}

function getRuleTypeLabel(type?: string): string {
  const map: Record<string, string> = {
    MATERIAL_CHECK: t('mes.pokayoke.typeMaterialCheck'),
    SEQUENCE_CHECK: t('mes.pokayoke.typeSequenceCheck'),
    PARAMETER_CHECK: t('mes.pokayoke.typeParameterCheck'),
    STATION_CHECK: t('mes.pokayoke.typeStationCheck'),
  }
  return map[type || ''] || type || '-'
}

function getRuleTypeTag(type?: string): string {
  const map: Record<string, string> = {
    MATERIAL_CHECK: 'danger',
    SEQUENCE_CHECK: 'warning',
    PARAMETER_CHECK: 'success',
    STATION_CHECK: 'info',
  }
  return map[type || ''] || ''
}

function getStatusLabel(status?: string): string {
  const map: Record<string, string> = {
    DRAFT: t('mes.pokayoke.statusDraft'),
    ACTIVE: t('mes.pokayoke.statusActive'),
    INACTIVE: t('mes.pokayoke.statusInactive'),
  }
  return map[status || ''] || status || '-'
}

function getStatusTag(status?: string): string {
  const map: Record<string, string> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    INACTIVE: 'warning',
  }
  return map[status || ''] || ''
}

function getActionTypeLabel(type?: string): string {
  const map: Record<string, string> = {
    BLOCK: t('mes.pokayoke.actionBlock'),
    WARNING: t('mes.pokayoke.actionWarning'),
    LOG: t('mes.pokayoke.actionLog'),
  }
  return map[type || ''] || type || '-'
}

function formatDateTime(dateTime?: string): string {
  if (!dateTime) return '-'
  return new Date(dateTime).toLocaleString()
}
</script>

<style scoped>
.pokayoke-rule-detail-container {
  padding: 16px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.code-block {
  background: #f5f7fa;
  padding: 8px 12px;
  border-radius: 4px;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: monospace;
}
</style>