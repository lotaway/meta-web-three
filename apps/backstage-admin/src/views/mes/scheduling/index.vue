<template>
  <div class="app-container">
    <div class="filter-container">
      <el-input v-model="listQuery.scheduleNo" :placeholder="t('mes.scheduling.scheduleNoPlaceholder')"
        class="filter-item" style="width: 180px" @keyup.enter="handleFilter" />
      <el-select v-model="listQuery.status" :placeholder="t('mes.scheduling.statusPlaceholder')"
        clearable class="filter-item" style="width: 140px">
        <el-option v-for="item in statusOptions" :key="item.value"
          :label="t(item.label)" :value="item.value" />
      </el-select>
      <el-select v-model="listQuery.workshopId" :placeholder="t('mes.scheduling.workshopPlaceholder')"
        clearable class="filter-item" style="width: 140px">
        <el-option label="Workshop-001" value="Workshop-001" />
      </el-select>
      <el-button class="filter-item" type="primary" :icon="Search" @click="handleFilter">
        {{ t('common.query') }}
      </el-button>
      <el-button class="filter-item" type="primary" :icon="Plus" @click="handleCreate">
        {{ t('mes.scheduling.add') }}
      </el-button>
    </div>

    <div class="schedule-controls" style="margin-bottom: 16px; display: flex; gap: 8px; align-items: center;">
      <el-button type="success" :icon="Top" :loading="schedulingForward" @click="runForwardSchedule">
        {{ t('mes.scheduling.runForward') }}
      </el-button>
      <el-button type="warning" :icon="Bottom" :loading="schedulingBackward" @click="runBackwardSchedule">
        {{ t('mes.scheduling.runBackward') }}
      </el-button>
      <el-tag v-if="overdueOrders.length > 0" type="danger" effect="dark" style="margin-left: 8px;">
        {{ t('mes.scheduling.overdueAlert', { count: overdueOrders.length }) }}
      </el-tag>
    </div>

    <el-alert v-if="lastResult" :title="resultTitle" :type="resultType" show-icon closable
      style="margin-bottom: 16px;">
      <template #default>
        <div>{{ t('mes.scheduling.scheduledCount', { count: lastResult.scheduledCount }) }}</div>
        <div v-if="lastResult.conflicts && lastResult.conflicts.length > 0">
          <div v-for="(c, i) in lastResult.conflicts" :key="i" style="font-size: 13px; margin-top: 4px;">
            - {{ c.orderNo }} / {{ c.resourceCode }}: {{ c.description }}
          </div>
        </div>
      </template>
    </el-alert>

    <el-tabs v-model="activeTab">
      <el-tab-pane :label="t('mes.scheduling.orders')" name="orders">
        <el-table v-loading="listLoading" :data="orderList" border style="width: 100%">
          <el-table-column type="expand">
            <template #default="{ row }">
              <div v-if="row.operations && row.operations.length > 0" style="padding: 12px;">
                <el-table :data="row.operations" size="small" border>
                  <el-table-column label="#" prop="sequenceNo" width="50" />
                  <el-table-column :label="t('mes.scheduling.operationCode')" prop="operationCode" width="100" />
                  <el-table-column :label="t('mes.scheduling.operationName')" prop="operationName" min-width="100" />
                  <el-table-column :label="t('mes.scheduling.resourceCode')" prop="resourceCode" width="100" />
                  <el-table-column :label="t('mes.scheduling.resourceName')" prop="resourceName" width="100" />
                  <el-table-column :label="t('mes.scheduling.setupTime')" prop="setupTimeMinutes" width="80" />
                  <el-table-column :label="t('mes.scheduling.processingTime')" prop="processingTimeMinutes" width="80" />
                  <el-table-column :label="t('mes.scheduling.teardownTime')" prop="teardownTimeMinutes" width="80" />
                  <el-table-column :label="t('mes.scheduling.opStatus')" prop="status" width="100">
                    <template #default="{ row: op }">
                      <el-tag :type="getOpStatusType(op.status)" size="small">
                        {{ t(`mes.scheduling.opStatus${op.status}`) }}
                      </el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column :label="t('mes.scheduling.scheduledStart')" width="140">
                    <template #default="{ row: op }">
                      {{ op.scheduledStartTime ? formatDateTime(op.scheduledStartTime) : '-' }}
                    </template>
                  </el-table-column>
                  <el-table-column :label="t('mes.scheduling.scheduledEnd')" width="140">
                    <template #default="{ row: op }">
                      {{ op.scheduledEndTime ? formatDateTime(op.scheduledEndTime) : '-' }}
                    </template>
                  </el-table-column>
                </el-table>
              </div>
              <el-empty v-else :description="t('mes.scheduling.noOperations')" />
            </template>
          </el-table-column>
          <el-table-column :label="t('common.id')" prop="id" width="60" />
          <el-table-column :label="t('mes.scheduling.scheduleNo')" prop="scheduleNo" width="120" />
          <el-table-column :label="t('mes.scheduling.orderNo')" prop="orderNo" width="120" />
          <el-table-column :label="t('mes.scheduling.productName')" prop="productName" min-width="100" />
          <el-table-column :label="t('mes.scheduling.quantity')" prop="quantity" width="80" />
          <el-table-column :label="t('mes.scheduling.completedQty')" prop="completedQuantity" width="80" />
          <el-table-column :label="t('mes.scheduling.priority')" prop="priority" width="80">
            <template #default="{ row }">
              <el-tag :type="getPriorityType(row.priority)" size="small">
                {{ t(`mes.scheduling.priority${row.priority}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.status')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="getStatusType(row.status)">
                {{ t(`mes.scheduling.status${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.dueDate')" width="100">
            <template #default="{ row }">
              {{ row.dueDate ? formatDate(row.dueDate) : '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.scheduledStart')" width="120">
            <template #default="{ row }">
              {{ row.scheduledStartTime ? formatDateTime(row.scheduledStartTime) : '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.scheduledEnd')" width="120">
            <template #default="{ row }">
              {{ row.scheduledEndTime ? formatDateTime(row.scheduledEndTime) : '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('common.operations')" width="260" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click="handleView(row)">{{ t('common.detail') }}</el-button>
              <el-button v-if="row.status === 'PENDING'" type="primary" link @click="handleAddOps(row)">
                {{ t('mes.scheduling.addOps') }}
              </el-button>
              <el-button v-if="row.status === 'SCHEDULED'" type="success" link @click="handleStart(row)">
                {{ t('mes.scheduling.start') }}
              </el-button>
              <el-button v-if="row.status === 'IN_PROGRESS'" type="success" link @click="handleComplete(row)">
                {{ t('mes.scheduling.complete') }}
              </el-button>
              <el-button v-if="row.status !== 'COMPLETED' && row.status !== 'CANCELLED'"
                type="danger" link @click="handleCancel(row)">
                {{ t('mes.scheduling.cancel') }}
              </el-button>
            </template>
          </el-table-column>
        </el-table>

        <el-pagination v-model:current-page="listQuery.page" v-model:page-size="listQuery.pageSize"
          :total="total" :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="getList" @current-change="getList" />
      </el-tab-pane>

      <el-tab-pane :label="t('mes.scheduling.resources')" name="resources">
        <el-button class="filter-item" type="primary" :icon="Plus" @click="handleCreateResource" style="margin-bottom: 12px;">
          {{ t('mes.scheduling.addResource') }}
        </el-button>
        <el-table v-loading="resourceLoading" :data="resourceList" border style="width: 100%">
          <el-table-column :label="t('common.id')" prop="id" width="60" />
          <el-table-column :label="t('mes.scheduling.resourceCode')" prop="resourceCode" width="120" />
          <el-table-column :label="t('mes.scheduling.resourceName')" prop="resourceName" min-width="120" />
          <el-table-column :label="t('mes.scheduling.resourceType')" prop="resourceType" width="100">
            <template #default="{ row }">
              {{ t(`mes.scheduling.resourceType${row.resourceType}`) }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.resourceStatus')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="getResourceStatusType(row.status)" size="small">
                {{ t(`mes.scheduling.resourceStatus${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.capacityPerShift')" prop="capacityPerShift" width="100" />
          <el-table-column :label="t('mes.scheduling.workshop')" prop="workshopId" width="120" />
          <el-table-column :label="t('common.operations')" width="160" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click="handleEditResource(row)">{{ t('common.edit') }}</el-button>
              <el-button type="danger" link @click="handleDeleteResource(row)">{{ t('common.delete') }}</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <el-dialog :title="t('mes.scheduling.addOps')" v-model="opsDialogVisible" width="700px">
      <el-form :model="opsForm" label-width="120px">
        <div v-for="(op, idx) in opsForm.operations" :key="idx"
          style="border: 1px solid #eee; padding: 12px; margin-bottom: 8px; border-radius: 4px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <strong>{{ t('mes.scheduling.operation') }} {{ idx + 1 }}</strong>
            <el-button type="danger" link @click="removeOperation(idx)">{{ t('common.delete') }}</el-button>
          </div>
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item :label="t('mes.scheduling.operationCode')" :required="true">
                <el-input v-model="op.operationCode" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item :label="t('mes.scheduling.operationName')">
                <el-input v-model="op.operationName" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-row :gutter="12">
            <el-col :span="12">
              <el-form-item :label="t('mes.scheduling.resourceCode')" :required="true">
                <el-input v-model="op.resourceCode" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item :label="t('mes.scheduling.resourceName')">
                <el-input v-model="op.resourceName" />
              </el-form-item>
            </el-col>
          </el-row>
          <el-row :gutter="12">
            <el-col :span="8">
              <el-form-item :label="t('mes.scheduling.setupTime')">
                <el-input-number v-model="op.setupTimeMinutes" :min="0" style="width: 100%" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item :label="t('mes.scheduling.processingTime')" :required="true">
                <el-input-number v-model="op.processingTimeMinutes" :min="1" style="width: 100%" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item :label="t('mes.scheduling.teardownTime')">
                <el-input-number v-model="op.teardownTimeMinutes" :min="0" style="width: 100%" />
              </el-form-item>
            </el-col>
          </el-row>
        </div>
        <el-button type="primary" :icon="Plus" @click="addOperation">
          {{ t('mes.scheduling.addOperation') }}
        </el-button>
      </el-form>
      <template #footer>
        <el-button @click="opsDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="opsSubmitting" @click="submitOperations">
          {{ t('common.confirm') }}
        </el-button>
      </template>
    </el-dialog>

    <el-dialog :title="t('mes.scheduling.createResource')" v-model="resourceDialogVisible" width="500px">
      <el-form :model="resourceForm" label-width="120px">
        <el-form-item :label="t('mes.scheduling.resourceCode')" :required="true">
          <el-input v-model="resourceForm.resourceCode" />
        </el-form-item>
        <el-form-item :label="t('mes.scheduling.resourceName')" :required="true">
          <el-input v-model="resourceForm.resourceName" />
        </el-form-item>
        <el-form-item :label="t('mes.scheduling.resourceType')" :required="true">
          <el-select v-model="resourceForm.resourceType" style="width: 100%">
            <el-option v-for="rt in resourceTypeOptions" :key="rt.value" :label="rt.label" :value="rt.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.scheduling.workshop')" :required="true">
          <el-input v-model="resourceForm.workshopId" />
        </el-form-item>
        <el-form-item :label="t('mes.scheduling.capacityPerShift')">
          <el-input-number v-model="resourceForm.capacityPerShift" :min="0" style="width: 100%" />
        </el-form-item>
        <el-form-item :label="t('mes.scheduling.description')">
          <el-input v-model="resourceForm.description" type="textarea" :rows="2" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="resourceDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="resourceSubmitting" @click="submitResource">
          {{ t('common.confirm') }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { Search, Plus, Top, Bottom } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import {
  getScheduleOrderListAPI,
  getOverdueOrdersAPI,
  deleteScheduleOrderAPI,
  runForwardScheduleAPI,
  runBackwardScheduleAPI,
  rescheduleOrderAPI,
  startOrderAPI,
  completeOrderAPI,
  cancelOrderAPI,
  addOperationsAPI,
  getResourceListAPI,
  createResourceAPI,
  deleteResourceAPI,
  type ScheduleOrder,
  type ScheduleResult,
  type ScheduleResource,
  type OperationRequest,
} from '@/apis/scheduling'

const { t } = useI18n()
const router = useRouter()

const activeTab = ref('orders')
const listLoading = ref(false)
const resourceLoading = ref(false)
const total = ref(0)

const listQuery = ref({
  scheduleNo: '',
  status: '',
  workshopId: '',
  page: 1,
  pageSize: 10,
})

const orderList = ref<ScheduleOrder[]>([])
const overdueOrders = ref<ScheduleOrder[]>([])
const resourceList = ref<ScheduleResource[]>([])
const lastResult = ref<ScheduleResult | null>(null)
const schedulingForward = ref(false)
const schedulingBackward = ref(false)

const statusOptions = [
  { value: 'PENDING', label: 'mes.scheduling.statusPENDING' },
  { value: 'SCHEDULED', label: 'mes.scheduling.statusSCHEDULED' },
  { value: 'IN_PROGRESS', label: 'mes.scheduling.statusIN_PROGRESS' },
  { value: 'COMPLETED', label: 'mes.scheduling.statusCOMPLETED' },
  { value: 'DELAYED', label: 'mes.scheduling.statusDELAYED' },
  { value: 'CANCELLED', label: 'mes.scheduling.statusCANCELLED' },
]

const resourceTypeOptions = [
  { value: 'EQUIPMENT', label: t('mes.scheduling.resourceTypeEQUIPMENT') },
  { value: 'WORK_CENTER', label: t('mes.scheduling.resourceTypeWORK_CENTER') },
  { value: 'LABOR', label: t('mes.scheduling.resourceTypeLABOR') },
  { value: 'TOOL', label: t('mes.scheduling.resourceTypeTOOL') },
]

const resultTitle = ref('')
const resultType = ref<'success' | 'warning' | 'error'>('success')

const opsDialogVisible = ref(false)
const opsForm = ref<{ operations: OperationRequest[]; orderId: number | null }>({
  operations: [],
  orderId: null,
})
const opsSubmitting = ref(false)

const resourceDialogVisible = ref(false)
const resourceForm = ref({
  resourceCode: '',
  resourceName: '',
  resourceType: 'EQUIPMENT',
  workshopId: 'Workshop-001',
  capacityPerShift: 0,
  description: '',
})
const resourceSubmitting = ref(false)

function getStatusType(status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'info', SCHEDULED: 'primary', IN_PROGRESS: 'warning',
    COMPLETED: 'success', DELAYED: 'danger', CANCELLED: 'info',
  }
  return map[status] || 'info'
}

function getPriorityType(p: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    LOW: 'info', NORMAL: 'primary', HIGH: 'warning', URGENT: 'danger',
  }
  return map[p] || 'info'
}

function getOpStatusType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'info', SCHEDULED: 'primary', IN_PROGRESS: 'warning',
    COMPLETED: 'success', BLOCKED: 'danger',
  }
  return map[s] || 'info'
}

function getResourceStatusType(s: string): 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    AVAILABLE: 'success', OCCUPIED: 'warning', MAINTENANCE: 'info', OFFLINE: 'danger',
  }
  return map[s] || 'info'
}

function formatDateTime(dateStr: string) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  return d.toLocaleString()
}

function formatDate(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleDateString()
}

async function getList() {
  listLoading.value = true
  try {
    const params: Record<string, unknown> = {}
    if (listQuery.value.scheduleNo) params.scheduleNo = listQuery.value.scheduleNo
    if (listQuery.value.status) params.status = listQuery.value.status
    if (listQuery.value.workshopId) params.workshopId = listQuery.value.workshopId
    const res = await getScheduleOrderListAPI(params)
    orderList.value = res.data || []
    total.value = orderList.value.length
  } catch {
    ElMessage.error(t('mes.scheduling.loadFailed'))
  } finally {
    listLoading.value = false
  }
}

async function loadOverdue() {
  try {
    const res = await getOverdueOrdersAPI()
    overdueOrders.value = res.data || []
  } catch {
    // silent
  }
}

async function loadResources() {
  resourceLoading.value = true
  try {
    const res = await getResourceListAPI()
    resourceList.value = res.data || []
  } catch {
    ElMessage.error(t('mes.scheduling.loadFailed'))
  } finally {
    resourceLoading.value = false
  }
}

function handleFilter() {
  listQuery.value.page = 1
  getList()
}

function handleCreate() {
  router.push({ name: 'schedulingForm' })
}

function handleView(row: ScheduleOrder) {
  router.push({ name: 'schedulingDetail', query: { id: row.id } })
}

function handleAddOps(row: ScheduleOrder) {
  opsForm.value = { operations: [], orderId: row.id! }
  opsDialogVisible.value = true
}

function addOperation() {
  opsForm.value.operations.push({
    operationCode: '',
    operationName: '',
    resourceCode: '',
    resourceName: '',
    setupTimeMinutes: 0,
    processingTimeMinutes: 10,
    teardownTimeMinutes: 0,
  })
}

function removeOperation(idx: number) {
  opsForm.value.operations.splice(idx, 1)
}

async function submitOperations() {
  if (!opsForm.value.orderId || opsForm.value.operations.length === 0) {
    ElMessage.warning(t('mes.scheduling.noOperationsWarning'))
    return
  }
  opsSubmitting.value = true
  try {
    await addOperationsAPI(opsForm.value.orderId, opsForm.value.operations)
    ElMessage.success(t('mes.scheduling.opsAdded'))
    opsDialogVisible.value = false
    getList()
  } catch {
    ElMessage.error(t('mes.scheduling.opsAddFailed'))
  } finally {
    opsSubmitting.value = false
  }
}

async function runForwardSchedule() {
  const workshopId = listQuery.value.workshopId || 'Workshop-001'
  schedulingForward.value = true
  try {
    const res = await runForwardScheduleAPI(workshopId)
    lastResult.value = res.data
    resultTitle.value = res.data.status === 'SUCCESS'
      ? t('mes.scheduling.scheduleSuccess') : res.data.status === 'PARTIAL'
        ? t('mes.scheduling.schedulePartial') : t('mes.scheduling.scheduleFailed')
    resultType.value = res.data.status === 'SUCCESS' ? 'success'
      : res.data.status === 'PARTIAL' ? 'warning' : 'error'
    getList()
    loadResources()
  } catch {
    ElMessage.error(t('mes.scheduling.scheduleRunFailed'))
  } finally {
    schedulingForward.value = false
  }
}

async function runBackwardSchedule() {
  const workshopId = listQuery.value.workshopId || 'Workshop-001'
  schedulingBackward.value = true
  try {
    const res = await runBackwardScheduleAPI(workshopId)
    lastResult.value = res.data
    resultTitle.value = res.data.status === 'SUCCESS'
      ? t('mes.scheduling.scheduleSuccess') : res.data.status === 'PARTIAL'
        ? t('mes.scheduling.schedulePartial') : t('mes.scheduling.scheduleFailed')
    resultType.value = res.data.status === 'SUCCESS' ? 'success'
      : res.data.status === 'PARTIAL' ? 'warning' : 'error'
    getList()
    loadResources()
  } catch {
    ElMessage.error(t('mes.scheduling.scheduleRunFailed'))
  } finally {
    schedulingBackward.value = false
  }
}

async function handleStart(row: ScheduleOrder) {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmStart'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'info' })
    await startOrderAPI(row.id!)
    ElMessage.success(t('mes.scheduling.startSuccess'))
    getList()
  } catch { /* cancel */ }
}

async function handleComplete(row: ScheduleOrder) {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmComplete'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'info' })
    await completeOrderAPI(row.id!)
    ElMessage.success(t('mes.scheduling.completeSuccess'))
    getList()
  } catch { /* cancel */ }
}

async function handleCancel(row: ScheduleOrder) {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmCancel'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await cancelOrderAPI(row.id!)
    ElMessage.success(t('mes.scheduling.cancelSuccess'))
    getList()
  } catch { /* cancel */ }
}

function handleCreateResource() {
  resourceForm.value = {
    resourceCode: '',
    resourceName: '',
    resourceType: 'EQUIPMENT',
    workshopId: 'Workshop-001',
    capacityPerShift: 0,
    description: '',
  }
  resourceDialogVisible.value = true
}

function handleEditResource(row: ScheduleResource) {
  ElMessage.info('Edit resource: feature coming soon')
}

async function handleDeleteResource(row: ScheduleResource) {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmDeleteResource'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await deleteResourceAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    loadResources()
  } catch { /* cancel */ }
}

async function submitResource() {
  if (!resourceForm.value.resourceCode || !resourceForm.value.resourceName) {
    ElMessage.warning(t('mes.scheduling.resourceRequired'))
    return
  }
  resourceSubmitting.value = true
  try {
    await createResourceAPI(resourceForm.value)
    ElMessage.success(t('mes.scheduling.createResourceSuccess'))
    resourceDialogVisible.value = false
    loadResources()
  } catch {
    ElMessage.error(t('common.submitFailed'))
  } finally {
    resourceSubmitting.value = false
  }
}

onMounted(() => {
  getList()
  loadOverdue()
  loadResources()
})
</script>
