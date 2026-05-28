<template>
  <div class="pokayoke-rule-detail-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>规则详情</span>
          <div>
            <el-button @click="handleBack">返回</el-button>
            <el-button type="primary" @click="handleEdit">编辑</el-button>
          </div>
        </div>
      </template>

      <div v-loading="ruleStore.loading">
        <el-descriptions :column="2" border v-if="ruleStore.currentRule">
          <el-descriptions-item label="ID">
            {{ ruleStore.currentRule.id }}
          </el-descriptions-item>
          <el-descriptions-item label="规则编码">
            {{ ruleStore.currentRule.ruleCode }}
          </el-descriptions-item>
          <el-descriptions-item label="规则名称">
            {{ ruleStore.currentRule.ruleName }}
          </el-descriptions-item>
          <el-descriptions-item label="规则类型">
            <el-tag :type="getRuleTypeTag(ruleStore.currentRule.ruleType)">
              {{ getRuleTypeLabel(ruleStore.currentRule.ruleType) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusTag(ruleStore.currentRule.status)">
              {{ getStatusLabel(ruleStore.currentRule.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="优先级">
            {{ ruleStore.currentRule.priority }}
          </el-descriptions-item>
          <el-descriptions-item label="工位">
            {{ ruleStore.currentRule.workstationId || '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="错误提示">
            {{ ruleStore.currentRule.errorMessage || '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="条件表达式" :span="2">
            <pre class="code-block">{{ ruleStore.currentRule.conditionExpression || '-' }}</pre>
          </el-descriptions-item>
          <el-descriptions-item label="动作类型">
            {{ getActionTypeLabel(ruleStore.currentRule.actionType) }}
          </el-descriptions-item>
          <el-descriptions-item label="动作配置">
            <pre class="code-block">{{ ruleStore.currentRule.actionConfig || '-' }}</pre>
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ formatDateTime(ruleStore.currentRule.createdAt) }}
          </el-descriptions-item>
          <el-descriptions-item label="更新时间">
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
import { usePokayokeRuleStore } from '@/stores/pokayokeRule'

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
    MATERIAL_CHECK: '物料检查',
    SEQUENCE_CHECK: '顺序检查',
    PARAMETER_CHECK: '参数检查',
    STATION_CHECK: '工位检查',
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
    DRAFT: '草稿',
    ACTIVE: '激活',
    INACTIVE: '停用',
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
    BLOCK: '阻止操作',
    WARNING: '警告',
    LOG: '记录日志',
  }
  return map[type || ''] || type || '-'
}

function formatDateTime(dateTime?: string): string {
  if (!dateTime) return '-'
  return new Date(dateTime).toLocaleString('zh-CN')
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