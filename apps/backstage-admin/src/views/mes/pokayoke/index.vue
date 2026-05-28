<template>
  <div class="pokayoke-rule-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('pokayoke.title') }}</span>
          <el-button type="primary" @click="handleCreate">{{ t('pokayoke.add') }}</el-button>
        </div>
      </template>

      <el-form :inline="true" :model="queryParams" class="search-form">
        <el-form-item :label="t('pokayoke.ruleType')">
          <el-select v-model="queryParams.ruleType" :placeholder="t('common.selectPlaceholder')" clearable>
            <el-option :label="t('pokayoke.typeMaterialCheck')" value="MATERIAL_CHECK" />
            <el-option :label="t('pokayoke.typeSequenceCheck')" value="SEQUENCE_CHECK" />
            <el-option :label="t('pokayoke.typeParameterCheck')" value="PARAMETER_CHECK" />
            <el-option :label="t('pokayoke.typeStationCheck')" value="STATION_CHECK" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('pokayoke.status')">
          <el-select v-model="queryParams.status" :placeholder="t('common.selectPlaceholder')" clearable>
            <el-option :label="t('pokayoke.statusDraft')" value="DRAFT" />
            <el-option :label="t('pokayoke.statusActive')" value="ACTIVE" />
            <el-option :label="t('pokayoke.statusInactive')" value="INACTIVE" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>

      <el-table :data="ruleStore.rules" v-loading="ruleStore.loading" border>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="ruleCode" :label="t('pokayoke.ruleCode')" width="120" />
        <el-table-column prop="ruleName" :label="t('pokayoke.ruleName')" min-width="150" />
        <el-table-column prop="ruleType" :label="t('pokayoke.ruleType')" width="120">
          <template #default="{ row }">
            <el-tag :type="getRuleTypeTag(row.ruleType)">
              {{ getRuleTypeLabel(row.ruleType) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('pokayoke.status')" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTag(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="priority" :label="t('pokayoke.priority')" width="80" />
        <el-table-column prop="workstationId" :label="t('pokayoke.workstationId')" width="100" />
        <el-table-column prop="conditionExpression" :label="t('pokayoke.conditionExpression')" min-width="200" show-overflow-tooltip />
        <el-table-column prop="errorMessage" :label="t('pokayoke.errorMessage')" min-width="150" show-overflow-tooltip />
        <el-table-column :label="t('common.operations')" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="primary" @click="handleDetail(row)">{{ t('common.detail') }}</el-button>
            <el-button link v-if="row.status === 'DRAFT' || row.status === 'INACTIVE'" 
              type="success" @click="handleActivate(row)">{{ t('pokayoke.activate') }}</el-button>
            <el-button link v-if="row.status === 'ACTIVE'" 
              type="warning" @click="handleDeactivate(row)">{{ t('pokayoke.deactivate') }}</el-button>
            <el-button link type="danger" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { usePokayokeRuleStore } from '@/stores/pokayokeRule'
import type { PokayokeRule, RuleStatus, RuleType } from '@/apis/pokayokeRule'

const { t } = useI18n()
const router = useRouter()
const ruleStore = usePokayokeRuleStore()

const queryParams = reactive<{
  ruleType?: RuleType
  status?: RuleStatus
}>({})

onMounted(() => {
  ruleStore.fetchRules()
})

function handleQuery() {
  ruleStore.fetchRules(queryParams)
}

function handleReset() {
  queryParams.ruleType = undefined
  queryParams.status = undefined
  ruleStore.fetchRules()
}

function handleCreate() {
  router.push('/mes/pokayoke/form')
}

function handleEdit(row: PokayokeRule) {
  router.push(`/mes/pokayoke/form?id=${row.id}`)
}

function handleDetail(row: PokayokeRule) {
  router.push(`/mes/pokayoke/detail?id=${row.id}`)
}

async function handleActivate(row: PokayokeRule) {
  try {
    await ruleStore.activateRule(row.id!)
    ElMessage.success(t('pokayoke.activateSuccess'))
    ruleStore.fetchRules(queryParams)
  } catch (error) {
    ElMessage.error(t('pokayoke.activateFailed'))
  }
}

async function handleDeactivate(row: PokayokeRule) {
  try {
    await ruleStore.deactivateRule(row.id!)
    ElMessage.success(t('pokayoke.deactivateSuccess'))
    ruleStore.fetchRules(queryParams)
  } catch (error) {
    ElMessage.error(t('pokayoke.deactivateFailed'))
  }
}

async function handleDelete(row: PokayokeRule) {
  try {
    await ElMessageBox.confirm(t('pokayoke.confirmDelete'), t('common.warning'), {
      confirmButtonText: t('common.confirmText'),
      cancelButtonText: t('common.cancelText'),
      type: 'warning',
    })
    await ruleStore.deleteRule(row.id!)
    ElMessage.success(t('message.deleteSuccess'))
  } catch (error) {
    ElMessage.error(t('message.deleteFailed'))
  }
}

function getRuleTypeLabel(type?: string): string {
  const map: Record<string, string> = {
    MATERIAL_CHECK: t('pokayoke.typeMaterialCheck'),
    SEQUENCE_CHECK: t('pokayoke.typeSequenceCheck'),
    PARAMETER_CHECK: t('pokayoke.typeParameterCheck'),
    STATION_CHECK: t('pokayoke.typeStationCheck'),
  }
  return map[type || ''] || type || ''
}

function getRuleTypeTag(type?: string): 'success' | 'warning' | 'danger' | 'info' | undefined {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | undefined> = {
    MATERIAL_CHECK: 'danger',
    SEQUENCE_CHECK: 'warning',
    PARAMETER_CHECK: 'success',
    STATION_CHECK: 'info',
  }
  return map[type || ''] || 'info'
}

function getStatusLabel(status?: string): string {
  const map: Record<string, string> = {
    DRAFT: t('pokayoke.statusDraft'),
    ACTIVE: t('pokayoke.statusActive'),
    INACTIVE: t('pokayoke.statusInactive'),
  }
  return map[status || ''] || status || ''
}

function getStatusTag(status?: string): 'success' | 'warning' | 'danger' | 'info' | undefined {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | undefined> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    INACTIVE: 'warning',
  }
  return map[status || ''] || 'info'
}
</script>

<style scoped>
.pokayoke-rule-container {
  padding: 16px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.search-form {
  margin-bottom: 16px;
}
</style>