<template>
  <div class="trigger-rule-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.triggerRule.ruleCode')">
          <el-input v-model="queryParams.ruleCode" :placeholder="t('mes.qc.triggerRule.ruleCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.triggerRule.ruleName')">
          <el-input v-model="queryParams.ruleName" :placeholder="t('mes.qc.triggerRule.ruleNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.triggerRule.triggerType')">
          <el-select v-model="queryParams.triggerType" :placeholder="t('mes.qc.triggerRule.triggerTypePlaceholder')" clearable>
            <el-option :label="t('mes.qc.triggerRule.triggerTypeAuto')" value="AUTO" />
            <el-option :label="t('mes.qc.triggerRule.triggerTypeManual')" value="MANUAL" />
            <el-option :label="t('mes.qc.triggerRule.triggerTypeScheduled')" value="SCHEDULED" />
            <el-option :label="t('mes.qc.triggerRule.triggerTypeEventBased')" value="EVENT_BASED" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.qc.triggerRule.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.qc.triggerRule.statusPlaceholder')" clearable>
            <el-option :label="t('mes.qc.triggerRule.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.triggerRule.statusInactive')" value="INACTIVE" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
          <el-button @click="resetQuery">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.triggerRule.add') }}</el-button>
      </div>

      <el-table :data="ruleList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.triggerRule.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.triggerRule.ruleCode')" prop="ruleCode" width="120" />
        <el-table-column :label="t('mes.qc.triggerRule.ruleName')" prop="ruleName" width="150" />
        <el-table-column :label="t('mes.qc.triggerRule.triggerType')" prop="triggerType" width="120">
          <template #default="{ row }">
            {{ getTriggerTypeText(row.triggerType) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.triggerRule.targetEntity')" prop="targetEntity" width="120" />
        <el-table-column :label="t('mes.qc.triggerRule.priority')" prop="priority" width="80" />
        <el-table-column :label="t('mes.qc.triggerRule.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'">
              {{ row.status === 'ACTIVE' ? t('mes.qc.triggerRule.statusActive') : t('mes.qc.triggerRule.statusInactive') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.triggerRule.description')" prop="description" />
        <el-table-column :label="t('common.operation')" fixed="right" width="180">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleEnable(row)" v-if="row.status === 'INACTIVE'">{{ t('mes.qc.triggerRule.enable') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDisable(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.qc.triggerRule.disable') }}</el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="queryParams.pageNum"
        v-model:page-size="queryParams.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { QcTriggerRule, TriggerType } from '@/apis/qc'
import {
  getTriggerRuleListAPI,
  deleteTriggerRuleAPI,
  enableTriggerRuleAPI,
  disableTriggerRuleAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const ruleList = ref<QcTriggerRule[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  ruleCode: '',
  ruleName: '',
  triggerType: '' as TriggerType | '',
  status: ''
})

const getTriggerTypeText = (type: string) => {
  const textMap: Record<string, string> = {
    AUTO: t('mes.qc.triggerRule.triggerTypeAuto'),
    MANUAL: t('mes.qc.triggerRule.triggerTypeManual'),
    SCHEDULED: t('mes.qc.triggerRule.triggerTypeScheduled'),
    EVENT_BASED: t('mes.qc.triggerRule.triggerTypeEventBased')
  }
  return textMap[type] || type
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getTriggerRuleListAPI()
    ruleList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.triggerRule.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.ruleCode = ''
  queryParams.ruleName = ''
  queryParams.triggerType = ''
  queryParams.status = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/triggerRule/form' })
}

const handleView = (row: QcTriggerRule) => {
  router.push({ path: '/mes/triggerRule/detail', query: { id: row.id } })
}

const handleEdit = (row: QcTriggerRule) => {
  router.push({ path: '/mes/triggerRule/form', query: { id: row.id } })
}

const handleEnable = async (row: QcTriggerRule) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.triggerRule.confirmEnable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await enableTriggerRuleAPI(row.id!)
    ElMessage.success(t('mes.qc.triggerRule.enableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.triggerRule.enableFailed'))
    }
  }
}

const handleDisable = async (row: QcTriggerRule) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.triggerRule.confirmDisable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await disableTriggerRuleAPI(row.id!)
    ElMessage.success(t('mes.qc.triggerRule.disableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.triggerRule.disableFailed'))
    }
  }
}

const handleDelete = async (row: QcTriggerRule) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.triggerRule.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteTriggerRuleAPI(row.id!)
    ElMessage.success(t('mes.qc.triggerRule.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.triggerRule.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.trigger-rule-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>