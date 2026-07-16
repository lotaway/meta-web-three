<template>
  <div class="crm-campaigns">
    <h2>{{ t('crm.campaign.title') }}</h2>
    <el-card class="search-bar">
      <el-form :inline="true" :model="queryParams">
        <el-form-item>
          <el-select v-model="queryParams.status" :placeholder="t('crm.campaign.status')" clearable>
            <el-option v-for="(label, key) in campaignStatusOptions" :key="key" :label="label" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.type" :placeholder="t('crm.campaign.type')" clearable>
            <el-option v-for="(label, key) in campaignTypeOptions" :key="key" :label="label" :value="key" />
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
      <el-table :data="campaigns" stripe v-loading="loading">
        <el-table-column prop="name" :label="t('crm.campaign.name')" />
        <el-table-column prop="type" :label="t('crm.campaign.type')" width="100">
          <template #default="{ row }">
            <el-tag size="small">{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('crm.campaign.status')" width="90">
          <template #default="{ row }">
            <el-tag :type="campaignStatusType(row.status)" size="small">{{ t('crm.campaignStatus.' + row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="startDate" :label="t('crm.campaign.startDate')" width="100">
          <template #default="{ row }">{{ row.startDate?.slice(0, 10) }}</template>
        </el-table-column>
        <el-table-column prop="endDate" :label="t('crm.campaign.endDate')" width="100">
          <template #default="{ row }">{{ row.endDate?.slice(0, 10) }}</template>
        </el-table-column>
        <el-table-column prop="budget" :label="t('crm.campaign.budget')" width="100">
          <template #default="{ row }">{{ row.budget?.toLocaleString() }}</template>
        </el-table-column>
        <el-table-column prop="leadsGenerated" :label="t('crm.campaign.leadsGenerated')" width="100" />
        <el-table-column prop="convertedCustomers" :label="t('crm.campaign.convertedCustomers')" width="100" />
        <el-table-column :label="t('common.operations')" width="240" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="openEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button v-if="row.status === 'PLANNING'" size="small" type="success" link @click="handleStatusChange(row, 'ACTIVE')">{{ t('crm.campaign.start') }}</el-button>
            <el-button v-if="row.status === 'ACTIVE'" size="small" type="warning" link @click="handleStatusChange(row, 'PAUSED')">{{ t('crm.campaign.pause') }}</el-button>
            <el-button v-if="row.status === 'ACTIVE' || row.status === 'PAUSED'" size="small" type="success" link @click="handleStatusChange(row, 'COMPLETED')">{{ t('crm.campaign.complete') }}</el-button>
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
        @current-change="fetchCampaigns"
      />
    </el-card>

    <el-dialog v-model="dialogVisible" :title="isEdit ? t('crm.campaign.edit') : t('crm.campaign.add')" width="600px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-form-item :label="t('crm.campaign.name')" prop="name">
          <el-input v-model="form.name" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.campaign.type')" prop="type">
              <el-select v-model="form.type" style="width:100%">
                <el-option v-for="(label, key) in campaignTypeOptions" :key="key" :label="label" :value="key" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.campaign.budget')" prop="budget">
              <el-input-number v-model="form.budget" :min="0" style="width:100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.campaign.startDate')" prop="startDate">
              <el-date-picker v-model="form.startDate" type="date" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.campaign.endDate')" prop="endDate">
              <el-date-picker v-model="form.endDate" type="date" style="width:100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('crm.campaign.description')" prop="description">
          <el-input v-model="form.description" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleSave">{{ t('common.save') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { listCampaigns, createCampaign, updateCampaign, deleteCampaign } from '@/apis/crm'
import type { Campaign } from '@/apis/crm'

const campaigns = ref<Campaign[]>([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)

const queryParams = reactive({
  status: '',
  type: '',
})

const dialogVisible = ref(false)
const isEdit = ref(false)
const editingId = ref<number | undefined>()

const form = reactive({
  name: '',
  type: '',
  budget: 0,
  startDate: '',
  endDate: '',
  description: '',
})

const rules = {
  name: [{ required: true, message: t('crm.campaign.name') + t('common.requiredSuffix'), trigger: 'blur' }],
}

const campaignTypeOptions: Record<string, string> = {
  EMAIL: t('crm.campaign.typeEmail'),
  SOCIAL: t('crm.campaign.typeSocial'),
  EVENT: t('crm.campaign.typeEvent'),
  CONTENT: t('crm.campaign.typeContent'),
  PAID: t('crm.campaign.typePaid'),
  OTHER: t('crm.campaign.typeOther'),
}

const campaignStatusOptions: Record<string, string> = {
  PLANNING: t('crm.campaignStatus.PLANNING'),
  ACTIVE: t('crm.campaignStatus.ACTIVE'),
  PAUSED: t('crm.campaignStatus.PAUSED'),
  COMPLETED: t('crm.campaignStatus.COMPLETED'),
  CANCELLED: t('crm.campaignStatus.CANCELLED'),
}

const campaignStatusType = (s: string) => {
  const map: Record<string, string> = { PLANNING: 'info', ACTIVE: 'success', PAUSED: 'warning', COMPLETED: '', CANCELLED: 'danger' }
  return map[s] || 'info'
}

const fetchCampaigns = async () => {
  loading.value = true
  try {
    const res = await listCampaigns({ ...queryParams, page: page.value, pageSize: pageSize.value })
    campaigns.value = res.data.records
    total.value = res.data.total
  } catch (e) {
    console.error('Failed to fetch campaigns:', e)
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  page.value = 1
  fetchCampaigns()
}

const handleReset = () => {
  queryParams.status = ''
  queryParams.type = ''
  page.value = 1
  fetchCampaigns()
}

const openCreate = () => {
  isEdit.value = false
  editingId.value = undefined
  form.name = ''
  form.type = ''
  form.budget = 0
  form.startDate = ''
  form.endDate = ''
  form.description = ''
  dialogVisible.value = true
}

const openEdit = (row: Campaign) => {
  isEdit.value = true
  editingId.value = row.id
  form.name = row.name
  form.type = row.type
  form.budget = row.budget
  form.startDate = row.startDate
  form.endDate = row.endDate
  form.description = row.description
  dialogVisible.value = true
}

const handleSave = async () => {
  try {
    if (isEdit.value && editingId.value) {
      await updateCampaign({ id: editingId.value, ...form })
      ElMessage.success(t('message.updateSuccess'))
    } else {
      await createCampaign(form)
      ElMessage.success(t('message.createSuccess'))
    }
    dialogVisible.value = false
    fetchCampaigns()
  } catch (e) {
    console.error('Failed to save campaign:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleDelete = async (id: number) => {
  try {
    await deleteCampaign(id)
    ElMessage.success(t('message.deleteSuccess'))
    fetchCampaigns()
  } catch (e) {
    console.error('Failed to delete campaign:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleStatusChange = async (row: Campaign, status: string) => {
  try {
    await updateCampaign({ id: row.id, status })
    ElMessage.success(t('crm.campaign.statusUpdateSuccess'))
    fetchCampaigns()
  } catch (e) {
    console.error('Failed to update campaign status:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

onMounted(fetchCampaigns)
</script>

<style scoped>
.crm-campaigns { padding: 20px; }
.search-bar { margin-bottom: 16px; }
.table-header { margin-bottom: 16px; }
</style>
