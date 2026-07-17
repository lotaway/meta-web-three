<template>
  <div class="crm-opportunities">
    <h2>{{ t('crm.opportunity.title') }}</h2>

    <el-card class="pipeline-summary">
      <el-row :gutter="10">
        <el-col v-for="(item, index) in pipelineData" :key="index" :span="4">
          <div class="pipeline-stage-card">
            <div class="stage-count">{{ item.count }}</div>
            <div class="stage-label">{{ item.label }}</div>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <el-card class="search-bar">
      <el-form :inline="true" :model="queryParams">
        <el-form-item>
          <el-input v-model="queryParams.keywords" :placeholder="t('common.search')" clearable @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.stage" :placeholder="t('crm.opportunity.stage')" clearable>
            <el-option v-for="(label, key) in stageOptions" :key="key" :label="label" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="table-header">
        <el-button type="primary" @click="openCreate">{{ t('common.add') }}</el-button>
      </div>
      <el-table :data="opportunities" stripe v-loading="loading">
        <el-table-column prop="opportunityNo" :label="t('crm.opportunity.opportunityNo')" width="120" />
        <el-table-column prop="title" :label="t('crm.opportunity.title')" />
        <el-table-column prop="customerId" :label="t('crm.opportunity.customer')" />
        <el-table-column prop="amount" :label="t('crm.opportunity.amount')" width="100">
          <template #default="{ row }">{{ row.amount?.toLocaleString() }}</template>
        </el-table-column>
        <el-table-column prop="stage" :label="t('crm.opportunity.stage')" width="100">
          <template #default="{ row }">
            <el-tag :type="stageType(row.stage)" size="small">{{ t('crm.stage.' + row.stage) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="probability" :label="t('crm.opportunity.probability')" width="80">
          <template #default="{ row }">{{ row.probability }}%</template>
        </el-table-column>
        <el-table-column prop="expectedCloseDate" :label="t('crm.opportunity.expectedCloseDate')" width="100">
          <template #default="{ row }">{{ row.expectedCloseDate?.slice(0, 10) }}</template>
        </el-table-column>
        <el-table-column prop="assignedTo" :label="t('crm.opportunity.assignedTo')" />
        <el-table-column :label="t('common.operations')" width="280" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="openEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button v-if="row.stage !== 'CLOSED_WON' && row.stage !== 'CLOSED_LOST'" size="small" link @click="handleAdvance(row)">{{ t('crm.opportunity.advance') }}</el-button>
            <el-button size="small" type="success" link @click="openCloseWon(row)">{{ t('crm.opportunity.closeWon') }}</el-button>
            <el-button size="small" type="danger" link @click="openCloseLost(row)">{{ t('crm.opportunity.closeLost') }}</el-button>
            <el-popconfirm :title="t('common.confirm') + t('common.delete') + '?'" @confirm="handleDelete(row.id!)">
              <template #reference>
                <el-button size="small" type="danger" link>{{ t('common.delete') }}</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        v-model:current-page="page"
        :page-size="pageSize"
        :total="total"
        layout="total, prev, pager, next"
        @current-change="fetchOpportunities"
      />
    </el-card>

    <el-dialog v-model="dialogVisible" :title="isEdit ? t('crm.opportunity.edit') : t('crm.opportunity.add')" width="600px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-form-item :label="t('crm.opportunity.title')" prop="title">
          <el-input v-model="form.title" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.opportunity.amount')" prop="amount">
              <el-input-number v-model="form.amount" :min="0" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.opportunity.probability')" prop="probability">
              <el-input-number v-model="form.probability" :min="0" :max="100" style="width:100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('crm.opportunity.expectedCloseDate')" prop="expectedCloseDate">
          <el-date-picker v-model="form.expectedCloseDate" type="date" style="width:100%" />
        </el-form-item>
        <el-form-item :label="t('crm.opportunity.competitor')" prop="competitor">
          <el-input v-model="form.competitor" />
        </el-form-item>
        <el-form-item :label="t('crm.opportunity.description')" prop="description">
          <el-input v-model="form.description" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleSave">{{ t('common.save') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="closeWonVisible" :title="t('crm.opportunity.closeWon')" width="400px">
      <el-form>
        <el-form-item :label="t('crm.opportunity.actualCloseDate')">
          <el-date-picker v-model="closeWonDate" type="date" style="width:100%" />
        </el-form-item>
        <el-form-item :label="t('crm.opportunity.description')">
          <el-input v-model="closeWonDesc" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="closeWonVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="success" @click="handleCloseWon">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="closeLostVisible" :title="t('crm.opportunity.closeLost')" width="400px">
      <el-form>
        <el-form-item :label="t('crm.opportunity.closeLostReason')">
          <el-input v-model="closeLostReason" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="closeLostVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="danger" @click="handleCloseLost">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { listOpportunities, createOpportunity, updateOpportunity, deleteOpportunity, advanceStage, closeWon, closeLost, getPipelineSummary } from '@/apis/crm'
import type { Opportunity, PipelineSummary } from '@/apis/crm'

const opportunities = ref<Opportunity[]>([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)

const queryParams = reactive({
  keywords: '',
  stage: '',
})

const dialogVisible = ref(false)
const isEdit = ref(false)
const editingId = ref<number | undefined>()

const closeWonVisible = ref(false)
const closeLostVisible = ref(false)
const closeWonTarget = ref<Opportunity | null>(null)
const closeLostTarget = ref<Opportunity | null>(null)
const closeWonDate = ref('')
const closeWonDesc = ref('')
const closeLostReason = ref('')

const form = reactive({
  title: '',
  amount: 0,
  probability: 0,
  expectedCloseDate: '',
  competitor: '',
  description: '',
})

const rules = {
  title: [{ required: true, message: t('crm.opportunity.title') + t('common.requiredSuffix'), trigger: 'blur' }],
}

const stageOptions: Record<string, string> = {
  PROSPECTING: t('crm.stage.PROSPECTING'),
  QUALIFICATION: t('crm.stage.QUALIFICATION'),
  PROPOSAL: t('crm.stage.PROPOSAL'),
  NEGOTIATION: t('crm.stage.NEGOTIATION'),
  CLOSED_WON: t('crm.stage.CLOSED_WON'),
  CLOSED_LOST: t('crm.stage.CLOSED_LOST'),
}

const stageType = (s: string) => {
  const map: Record<string, string> = { PROSPECTING: 'info', QUALIFICATION: 'primary', PROPOSAL: 'warning', NEGOTIATION: 'danger', CLOSED_WON: 'success', CLOSED_LOST: 'danger' }
  return map[s] || 'info'
}

const pipelineData = ref<{ label: string; count: number }[]>([])

const fetchOpportunities = async () => {
  loading.value = true
  try {
    const res = await listOpportunities({ ...queryParams, page: page.value, pageSize: pageSize.value })
    opportunities.value = res.data.records
    total.value = res.data.total
  } catch (e) {
    console.error('Failed to fetch opportunities:', e)
    ElMessage.error('Failed to load opportunities')
  } finally {
    loading.value = false
  }
}

const fetchPipelineSummary = async () => {
  try {
    const res = await getPipelineSummary()
    pipelineData.value = res.data.stages.map((s, i) => ({
      label: t('crm.stage.' + s),
      count: res.data.counts[i],
    }))
  } catch (e) {
    console.error('Failed to fetch pipeline summary:', e)
  }
}

const handleSearch = () => {
  page.value = 1
  fetchOpportunities()
}

const handleReset = () => {
  queryParams.keywords = ''
  queryParams.stage = ''
  page.value = 1
  fetchOpportunities()
}

const openCreate = () => {
  isEdit.value = false
  editingId.value = undefined
  form.title = ''
  form.amount = 0
  form.probability = 0
  form.expectedCloseDate = ''
  form.competitor = ''
  form.description = ''
  dialogVisible.value = true
}

const openEdit = (row: Opportunity) => {
  isEdit.value = true
  editingId.value = row.id
  form.title = row.title
  form.amount = row.amount
  form.probability = row.probability
  form.expectedCloseDate = row.expectedCloseDate
  form.competitor = row.competitor
  form.description = row.description
  dialogVisible.value = true
}

const handleSave = async () => {
  try {
    if (isEdit.value && editingId.value) {
      await updateOpportunity({ id: editingId.value, ...form })
      ElMessage.success(t('message.updateSuccess'))
    } else {
      await createOpportunity(form)
      ElMessage.success(t('message.createSuccess'))
    }
    dialogVisible.value = false
    fetchOpportunities()
  } catch (e) {
    console.error('Failed to save opportunity:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleDelete = async (id: number) => {
  try {
    await deleteOpportunity(id)
    ElMessage.success(t('message.deleteSuccess'))
    fetchOpportunities()
  } catch (e) {
    console.error('Failed to delete opportunity:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleAdvance = async (row: Opportunity) => {
  try {
    await advanceStage(row.id!)
    ElMessage.success(t('crm.opportunity.advanceSuccess'))
    fetchOpportunities()
  } catch (e) {
    console.error('Failed to advance opportunity:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const openCloseWon = (row: Opportunity) => {
  closeWonTarget.value = row
  closeWonDate.value = ''
  closeWonDesc.value = ''
  closeWonVisible.value = true
}

const handleCloseWon = async () => {
  if (!closeWonTarget.value) return
  try {
    await closeWon(closeWonTarget.value.id!, { actualCloseDate: closeWonDate.value, description: closeWonDesc.value })
    ElMessage.success(t('crm.opportunity.closeWonSuccess'))
    closeWonVisible.value = false
    fetchOpportunities()
  } catch (e) {
    console.error('Failed to close won:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const openCloseLost = (row: Opportunity) => {
  closeLostTarget.value = row
  closeLostReason.value = ''
  closeLostVisible.value = true
}

const handleCloseLost = async () => {
  if (!closeLostTarget.value) return
  try {
    await closeLost(closeLostTarget.value.id!, { reason: closeLostReason.value })
    ElMessage.success(t('crm.opportunity.closeLostSuccess'))
    closeLostVisible.value = false
    fetchOpportunities()
  } catch (e) {
    console.error('Failed to close lost:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

onMounted(() => {
  fetchOpportunities()
  fetchPipelineSummary()
})
</script>

<style scoped>
.crm-opportunities { padding: 20px; }
.search-bar { margin-bottom: 16px; }
.pipeline-summary { margin-bottom: 16px; }
.pipeline-stage-card { text-align: center; padding: 8px; }
.stage-count { font-size: 24px; font-weight: bold; color: #409eff; }
.stage-label { font-size: 12px; color: #909399; margin-top: 4px; }
.table-header { margin-bottom: 16px; }
</style>
