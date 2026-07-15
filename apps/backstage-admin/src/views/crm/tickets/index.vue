<template>
  <div class="crm-tickets">
    <h2>{{ t('crm.ticket.title') }}</h2>
    <el-card class="search-bar">
      <el-form :inline="true" :model="queryParams">
        <el-form-item>
          <el-input v-model="queryParams.keywords" :placeholder="t('common.search')" clearable @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.status" :placeholder="t('crm.ticket.status')" clearable>
            <el-option v-for="(label, key) in ticketStatusOptions" :key="key" :label="label" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.priority" :placeholder="t('crm.ticket.priority')" clearable>
            <el-option v-for="(label, key) in priorityOptions" :key="key" :label="label" :value="key" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-select v-model="queryParams.type" :placeholder="t('crm.ticket.type')" clearable>
            <el-option v-for="(label, key) in ticketTypeOptions" :key="key" :label="label" :value="key" />
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
      <el-table :data="tickets" stripe v-loading="loading">
        <el-table-column prop="ticketNo" :label="t('crm.ticket.ticketNo')" width="120" />
        <el-table-column prop="title" :label="t('crm.ticket.title')" />
        <el-table-column prop="type" :label="t('crm.ticket.type')" width="80">
          <template #default="{ row }">
            <el-tag size="small">{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="priority" :label="t('crm.ticket.priority')" width="80">
          <template #default="{ row }">
            <el-tag :type="priorityType(row.priority)" size="small">{{ t('crm.ticketPriority.' + row.priority) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('crm.ticket.status')" width="90">
          <template #default="{ row }">
            <el-tag :type="ticketStatusType(row.status)" size="small">{{ t('crm.ticketStatus.' + row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="assignedTo" :label="t('crm.ticket.assignedTo')" />
        <el-table-column prop="createdAt" :label="t('crm.ticket.createdAt')" width="100">
          <template #default="{ row }">{{ row.createdAt?.slice(0, 10) }}</template>
        </el-table-column>
        <el-table-column :label="t('common.operations')" width="280" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="openEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button size="small" link @click="openAssign(row)">{{ t('crm.ticket.assign') }}</el-button>
            <el-button v-if="row.status === 'OPEN' || row.status === 'ASSIGNED'" size="small" type="primary" link @click="handleStatusUpdate(row, 'IN_PROGRESS')">{{ t('crm.ticket.start') }}</el-button>
            <el-button v-if="row.status === 'IN_PROGRESS'" size="small" type="success" link @click="openResolve(row)">{{ t('crm.ticket.resolve') }}</el-button>
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
        @current-change="fetchTickets"
      />
    </el-card>

    <el-dialog v-model="dialogVisible" :title="isEdit ? t('crm.ticket.edit') : t('crm.ticket.add')" width="600px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="100px">
        <el-form-item :label="t('crm.ticket.title')" prop="title">
          <el-input v-model="form.title" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.ticket.type')" prop="type">
              <el-select v-model="form.type" style="width:100%">
                <el-option v-for="(label, key) in ticketTypeOptions" :key="key" :label="label" :value="key" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.ticket.priority')" prop="priority">
              <el-select v-model="form.priority" style="width:100%">
                <el-option v-for="(label, key) in priorityOptions" :key="key" :label="label" :value="key" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('crm.ticket.description')" prop="description">
          <el-input v-model="form.description" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleSave">{{ t('common.save') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="assignVisible" :title="t('crm.ticket.assign')" width="400px">
      <el-form>
        <el-form-item :label="t('crm.ticket.assignedTo')">
          <el-input v-model="assignTo" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="assignVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleAssign">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="resolveVisible" :title="t('crm.ticket.resolve')" width="400px">
      <el-form>
        <el-form-item :label="t('crm.ticket.resolution')">
          <el-input v-model="resolutionText" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="resolveVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="success" @click="handleResolve">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { listTickets, createTicket, updateTicket, deleteTicket, assignTicket, updateTicketStatus } from '@/apis/crm'
import type { CustomerServiceTicket } from '@/apis/crm'

const tickets = ref<CustomerServiceTicket[]>([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)

const queryParams = reactive({
  keywords: '',
  status: '',
  priority: '',
  type: '',
})

const dialogVisible = ref(false)
const isEdit = ref(false)
const editingId = ref<number | undefined>()

const assignVisible = ref(false)
const assignTarget = ref<CustomerServiceTicket | null>(null)
const assignTo = ref('')

const resolveVisible = ref(false)
const resolveTarget = ref<CustomerServiceTicket | null>(null)
const resolutionText = ref('')

const form = reactive({
  title: '',
  type: '',
  priority: '',
  description: '',
})

const rules = {
  title: [{ required: true, message: t('crm.ticket.title') + t('common.requiredSuffix'), trigger: 'blur' }],
}

const ticketTypeOptions: Record<string, string> = {
  COMPLAINT: t('crm.ticket.typeComplaint'),
  INQUIRY: t('crm.ticket.typeInquiry'),
  AFTER_SALE: t('crm.ticket.typeAfterSale'),
  OTHER: t('crm.ticket.typeOther'),
}

const priorityOptions: Record<string, string> = {
  LOW: t('crm.ticketPriority.LOW'),
  MEDIUM: t('crm.ticketPriority.MEDIUM'),
  HIGH: t('crm.ticketPriority.HIGH'),
  URGENT: t('crm.ticketPriority.URGENT'),
}

const ticketStatusOptions: Record<string, string> = {
  OPEN: t('crm.ticketStatus.OPEN'),
  ASSIGNED: t('crm.ticketStatus.ASSIGNED'),
  IN_PROGRESS: t('crm.ticketStatus.IN_PROGRESS'),
  RESOLVED: t('crm.ticketStatus.RESOLVED'),
  CLOSED: t('crm.ticketStatus.CLOSED'),
}

const priorityType = (p: string) => {
  const map: Record<string, string> = { LOW: 'info', MEDIUM: '', HIGH: 'warning', URGENT: 'danger' }
  return map[p] || 'info'
}

const ticketStatusType = (s: string) => {
  const map: Record<string, string> = { OPEN: 'info', ASSIGNED: 'primary', IN_PROGRESS: 'warning', RESOLVED: 'success', CLOSED: '' }
  return map[s] || 'info'
}

const fetchTickets = async () => {
  loading.value = true
  try {
    const res = await listTickets({ ...queryParams, page: page.value, pageSize: pageSize.value })
    tickets.value = res.data.records
    total.value = res.data.total
  } catch {
    // silent
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  page.value = 1
  fetchTickets()
}

const handleReset = () => {
  queryParams.keywords = ''
  queryParams.status = ''
  queryParams.priority = ''
  queryParams.type = ''
  page.value = 1
  fetchTickets()
}

const openCreate = () => {
  isEdit.value = false
  editingId.value = undefined
  form.title = ''
  form.type = ''
  form.priority = ''
  form.description = ''
  dialogVisible.value = true
}

const openEdit = (row: CustomerServiceTicket) => {
  isEdit.value = true
  editingId.value = row.id
  form.title = row.title
  form.type = row.type
  form.priority = row.priority
  form.description = row.description
  dialogVisible.value = true
}

const handleSave = async () => {
  try {
    if (isEdit.value && editingId.value) {
      await updateTicket({ id: editingId.value, ...form })
      ElMessage.success(t('message.updateSuccess'))
    } else {
      await createTicket(form)
      ElMessage.success(t('message.createSuccess'))
    }
    dialogVisible.value = false
    fetchTickets()
  } catch {
    // silent
  }
}

const handleDelete = async (id: number) => {
  try {
    await deleteTicket(id)
    ElMessage.success(t('message.deleteSuccess'))
    fetchTickets()
  } catch {
    // silent
  }
}

const openAssign = (row: CustomerServiceTicket) => {
  assignTarget.value = row
  assignTo.value = ''
  assignVisible.value = true
}

const handleAssign = async () => {
  if (!assignTarget.value) return
  try {
    await assignTicket(assignTarget.value.id!, assignTo.value)
    ElMessage.success(t('crm.ticket.assignSuccess'))
    assignVisible.value = false
    fetchTickets()
  } catch {
    // silent
  }
}

const handleStatusUpdate = async (row: CustomerServiceTicket, status: string) => {
  try {
    await updateTicketStatus(row.id!, status)
    ElMessage.success(t('crm.ticket.statusUpdateSuccess'))
    fetchTickets()
  } catch {
    // silent
  }
}

const openResolve = (row: CustomerServiceTicket) => {
  resolveTarget.value = row
  resolutionText.value = ''
  resolveVisible.value = true
}

const handleResolve = async () => {
  if (!resolveTarget.value) return
  try {
    await updateTicketStatus(resolveTarget.value.id!, 'RESOLVED', resolutionText.value)
    ElMessage.success(t('crm.ticket.resolveSuccess'))
    resolveVisible.value = false
    fetchTickets()
  } catch {
    // silent
  }
}

onMounted(fetchTickets)
</script>

<style scoped>
.crm-tickets { padding: 20px; }
.search-bar { margin-bottom: 16px; }
.table-header { margin-bottom: 16px; }
</style>
