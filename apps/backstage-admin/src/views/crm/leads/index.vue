<template>
  <div class="crm-leads">
    <h2>{{ t('crm.lead.title') }}</h2>
    <el-card class="search-bar">
      <el-form :inline="true" :model="queryParams">
        <el-form-item>
          <el-input v-model="queryParams.keywords" :placeholder="t('common.search')" clearable @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.status" :placeholder="t('crm.lead.status')" clearable>
            <el-option v-for="(label, key) in leadStatusOptions" :key="key" :label="label" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.source" :placeholder="t('crm.lead.source')" clearable>
            <el-option v-for="(label, key) in sourceOptions" :key="key" :label="label" :value="key" />
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
      <el-table :data="leads" stripe v-loading="loading">
        <el-table-column prop="leadNo" :label="t('crm.lead.leadNo')" width="120" />
        <el-table-column prop="name" :label="t('crm.lead.name')" />
        <el-table-column prop="company" :label="t('crm.lead.company')" />
        <el-table-column prop="email" :label="t('crm.lead.email')" />
        <el-table-column prop="phone" :label="t('crm.lead.phone')" width="120" />
        <el-table-column prop="source" :label="t('crm.lead.source')" width="90">
          <template #default="{ row }">
            <el-tag size="small">{{ t('crm.source.' + row.source) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('crm.lead.status')" width="90">
          <template #default="{ row }">
            <el-tag :type="leadStatusType(row.status)" size="small">{{ t('crm.leadStatus.' + row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="score" :label="t('crm.lead.score')" width="70" />
        <el-table-column prop="assignedTo" :label="t('crm.lead.assignedTo')" />
        <el-table-column prop="createdAt" :label="t('crm.lead.createdAt')" width="100">
          <template #default="{ row }">{{ row.createdAt?.slice(0, 10) }}</template>
        </el-table-column>
        <el-table-column :label="t('common.operations')" width="240" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="openEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button size="small" type="success" link @click="handleConvert(row)">{{ t('crm.lead.convert') }}</el-button>
            <el-button size="small" type="warning" link @click="openDisqualify(row)">{{ t('crm.lead.disqualify') }}</el-button>
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
        @current-change="fetchLeads"
      />
    </el-card>

    <el-dialog v-model="dialogVisible" :title="isEdit ? t('crm.lead.edit') : t('crm.lead.add')" width="600px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="100px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.name')" prop="name">
              <el-input v-model="form.name" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.company')" prop="company">
              <el-input v-model="form.company" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.title')" prop="title">
              <el-input v-model="form.title" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.email')" prop="email">
              <el-input v-model="form.email" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.phone')" prop="phone">
              <el-input v-model="form.phone" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.mobile')" prop="mobile">
              <el-input v-model="form.mobile" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.source')" prop="source">
              <el-select v-model="form.source" style="width:100%">
                <el-option v-for="(label, key) in sourceOptions" :key="key" :label="label" :value="key" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.lead.industry')" prop="industry">
              <el-input v-model="form.industry" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('crm.lead.description')" prop="description">
          <el-input v-model="form.description" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleSave">{{ t('common.save') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="disqualifyVisible" :title="t('crm.lead.disqualify')" width="400px">
      <el-form>
        <el-form-item :label="t('crm.lead.disqualifyReason')">
          <el-input v-model="disqualifyReason" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="disqualifyVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleDisqualify">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { listLeads, createLead, updateLead, deleteLead, convertLead, disqualifyLead } from '@/apis/crm'
import type { Lead } from '@/apis/crm'

const leads = ref<Lead[]>([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)

const queryParams = reactive({
  keywords: '',
  status: '',
  source: '',
})

const dialogVisible = ref(false)
const disqualifyVisible = ref(false)
const isEdit = ref(false)
const editingId = ref<number | undefined>()
const disqualifyReason = ref('')
const disqualifyTarget = ref<Lead | null>(null)

const form = reactive({
  name: '',
  company: '',
  title: '',
  email: '',
  phone: '',
  mobile: '',
  source: '',
  industry: '',
  description: '',
})

const rules = {
  name: [{ required: true, message: t('crm.lead.name') + t('common.requiredSuffix'), trigger: 'blur' }],
  company: [{ required: true, message: t('crm.lead.company') + t('common.requiredSuffix'), trigger: 'blur' }],
}

const sourceOptions: Record<string, string> = {
  WEBSITE: t('crm.source.WEBSITE'),
  REFERRAL: t('crm.source.REFERRAL'),
  SOCIAL_MEDIA: t('crm.source.SOCIAL_MEDIA'),
  EVENT: t('crm.source.EVENT'),
  COLD_CALL: t('crm.source.COLD_CALL'),
  OTHER: t('crm.source.OTHER'),
}

const leadStatusOptions: Record<string, string> = {
  NEW: t('crm.leadStatus.NEW'),
  CONTACTED: t('crm.leadStatus.CONTACTED'),
  QUALIFIED: t('crm.leadStatus.QUALIFIED'),
  DISQUALIFIED: t('crm.leadStatus.DISQUALIFIED'),
  CONVERTED: t('crm.leadStatus.CONVERTED'),
}

const leadStatusType = (s: string) => {
  const map: Record<string, string> = { NEW: 'info', CONTACTED: 'primary', QUALIFIED: 'success', DISQUALIFIED: 'danger', CONVERTED: 'warning' }
  return map[s] || 'info'
}

const fetchLeads = async () => {
  loading.value = true
  try {
    const res = await listLeads({ ...queryParams, page: page.value, pageSize: pageSize.value })
    leads.value = res.data.records
    total.value = res.data.total
  } catch (e) {
    console.error('Failed to fetch leads:', e)
    ElMessage.error('Failed to load leads')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  page.value = 1
  fetchLeads()
}

const handleReset = () => {
  queryParams.keywords = ''
  queryParams.status = ''
  queryParams.source = ''
  page.value = 1
  fetchLeads()
}

const openCreate = () => {
  isEdit.value = false
  editingId.value = undefined
  form.name = ''
  form.company = ''
  form.title = ''
  form.email = ''
  form.phone = ''
  form.mobile = ''
  form.source = ''
  form.industry = ''
  form.description = ''
  dialogVisible.value = true
}

const openEdit = (row: Lead) => {
  isEdit.value = true
  editingId.value = row.id
  form.name = row.name
  form.company = row.company
  form.title = row.title
  form.email = row.email
  form.phone = row.phone
  form.mobile = row.mobile
  form.source = row.source
  form.industry = row.industry
  form.description = row.description
  dialogVisible.value = true
}

const handleSave = async () => {
  try {
    if (isEdit.value && editingId.value) {
      await updateLead({ id: editingId.value, ...form })
      ElMessage.success(t('message.updateSuccess'))
    } else {
      await createLead(form)
      ElMessage.success(t('message.createSuccess'))
    }
    dialogVisible.value = false
    fetchLeads()
  } catch (e) {
    console.error('Failed to save lead:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleDelete = async (id: number) => {
  try {
    await deleteLead(id)
    ElMessage.success(t('message.deleteSuccess'))
    fetchLeads()
  } catch (e) {
    console.error('Failed to delete lead:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleConvert = async (row: Lead) => {
  try {
    await convertLead(row.id!)
    ElMessage.success(t('crm.lead.convertSuccess'))
    fetchLeads()
  } catch (e) {
    console.error('Failed to convert lead:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const openDisqualify = (row: Lead) => {
  disqualifyTarget.value = row
  disqualifyReason.value = ''
  disqualifyVisible.value = true
}

const handleDisqualify = async () => {
  if (!disqualifyTarget.value) return
  try {
    await disqualifyLead(disqualifyTarget.value.id!, disqualifyReason.value)
    ElMessage.success(t('crm.lead.disqualifySuccess'))
    disqualifyVisible.value = false
    fetchLeads()
  } catch (e) {
    console.error('Failed to disqualify lead:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

onMounted(fetchLeads)
</script>

<style scoped>
.crm-leads { padding: 20px; }
.search-bar { margin-bottom: 16px; }
.table-header { margin-bottom: 16px; }
</style>
