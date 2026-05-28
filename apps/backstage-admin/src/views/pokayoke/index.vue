<template>
  <div class="pokayoke-rule-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>防错规则管理</span>
          <el-button type="primary" @click="handleCreate">新增规则</el-button>
        </div>
      </template>

      <el-form :inline="true" :model="queryParams" class="search-form">
        <el-form-item label="规则类型">
          <el-select v-model="queryParams.ruleType" placeholder="请选择" clearable>
            <el-option label="物料检查" value="MATERIAL_CHECK" />
            <el-option label="顺序检查" value="SEQUENCE_CHECK" />
            <el-option label="参数检查" value="PARAMETER_CHECK" />
            <el-option label="工位检查" value="STATION_CHECK" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="queryParams.status" placeholder="请选择" clearable>
            <el-option label="草稿" value="DRAFT" />
            <el-option label="激活" value="ACTIVE" />
            <el-option label="停用" value="INACTIVE" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">查询</el-button>
          <el-button @click="handleReset">重置</el-button>
        </el-form-item>
      </el-form>

      <el-table :data="ruleStore.rules" v-loading="ruleStore.loading" border>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="ruleCode" label="规则编码" width="120" />
        <el-table-column prop="ruleName" label="规则名称" min-width="150" />
        <el-table-column prop="ruleType" label="规则类型" width="120">
          <template #default="{ row }">
            <el-tag :type="getRuleTypeTag(row.ruleType)">
              {{ getRuleTypeLabel(row.ruleType) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTag(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="priority" label="优先级" width="80" />
        <el-table-column prop="workstationId" label="工位" width="100" />
        <el-table-column prop="conditionExpression" label="条件表达式" min-width="200" show-overflow-tooltip />
        <el-table-column prop="errorMessage" label="错误提示" min-width="150" show-overflow-tooltip />
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" @click="handleEdit(row)">编辑</el-button>
            <el-button link type="primary" @click="handleDetail(row)">详情</el-button>
            <el-button link v-if="row.status === 'DRAFT' || row.status === 'INACTIVE'" 
              type="success" @click="handleActivate(row)">激活</el-button>
            <el-button link v-if="row.status === 'ACTIVE'" 
              type="warning" @click="handleDeactivate(row)">停用</el-button>
            <el-button link type="danger" @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { usePokayokeRuleStore } from '@/stores/pokayokeRule'
import type { PokayokeRule, RuleStatus, RuleType } from '@/apis/pokayokeRule'

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
    ElMessage.success('激活成功')
    ruleStore.fetchRules(queryParams)
  } catch (error) {
    ElMessage.error('激活失败')
  }
}

async function handleDeactivate(row: PokayokeRule) {
  try {
    await ruleStore.deactivateRule(row.id!)
    ElMessage.success('停用成功')
    ruleStore.fetchRules(queryParams)
  } catch (error) {
    ElMessage.error('停用失败')
  }
}

async function handleDelete(row: PokayokeRule) {
  try {
    await ElMessageBox.confirm('确认删除该规则吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    })
    await ruleStore.deleteRule(row.id!)
    ElMessage.success('删除成功')
  } catch (error) {
    // cancelled
  }
}

function getRuleTypeLabel(type?: string): string {
  const map: Record<string, string> = {
    MATERIAL_CHECK: '物料检查',
    SEQUENCE_CHECK: '顺序检查',
    PARAMETER_CHECK: '参数检查',
    STATION_CHECK: '工位检查',
  }
  return map[type || ''] || type
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
  return map[status || ''] || status
}

function getStatusTag(status?: string): string {
  const map: Record<string, string> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    INACTIVE: 'warning',
  }
  return map[status || ''] || ''
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