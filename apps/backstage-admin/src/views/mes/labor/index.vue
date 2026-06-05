<template>
  <div class="app-container">
    <div class="filter-container">
      <el-input v-model="operatorQuery.operatorCode" :placeholder="t('mes.labor.operatorCodePlaceholder')"
        class="filter-item" style="width: 160px" @keyup.enter="handleFilter" />
      <el-select v-model="operatorQuery.department" :placeholder="t('mes.labor.departmentPlaceholder')"
        clearable class="filter-item" style="width: 140px">
        <el-option label="Production" value="Production" />
        <el-option label="Assembly" value="Assembly" />
        <el-option label="Quality" value="Quality" />
        <el-option label="Warehouse" value="Warehouse" />
      </el-select>
      <el-select v-model="operatorQuery.status" :placeholder="t('mes.labor.statusPlaceholder')"
        clearable class="filter-item" style="width: 130px">
        <el-option v-for="s in operatorStatusOptions" :key="s.value" :label="t(s.label)" :value="s.value" />
      </el-select>
      <el-button class="filter-item" type="primary" :icon="Search" @click="handleFilter">
        {{ t('common.query') }}
      </el-button>
      <el-button class="filter-item" type="primary" :icon="Plus" @click="handleCreate">
        {{ t('mes.labor.addOperator') }}
      </el-button>
    </div>

    <el-tabs v-model="activeTab">
      <el-tab-pane :label="t('mes.labor.operators')" name="operators">
        <el-table v-loading="operatorLoading" :data="operatorList" border @row-click="handleView">
          <el-table-column type="expand">
            <template #default="{ row }">
              <div v-if="row.skills && row.skills.length > 0" style="padding: 12px;">
                <el-table :data="row.skills" size="small" border>
                  <el-table-column :label="t('mes.labor.skillCode')" prop="skillCode" width="100" />
                  <el-table-column :label="t('mes.labor.skillName')" prop="skillName" min-width="100" />
                  <el-table-column :label="t('mes.labor.skillLevel')" prop="skillLevel" width="100">
                    <template #default="{ row: sk }">
                      <el-tag :type="getSkillLevelType(sk.skillLevel)" size="small">
                        {{ t(`mes.labor.skillLevel${sk.skillLevel}`) }}
                      </el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column :label="t('mes.labor.certified')" width="80">
                    <template #default="{ row: sk }">
                      <el-tag :type="sk.certified ? 'success' : 'info'" size="small">
                        {{ sk.certified ? t('common.yes') : t('common.no') }}
                      </el-tag>
                    </template>
                  </el-table-column>
                </el-table>
              </div>
              <div v-else style="padding: 12px; color: #999;">{{ t('mes.labor.noSkills') }}</div>
            </template>
          </el-table-column>
          <el-table-column :label="t('common.id')" prop="id" width="60" />
          <el-table-column :label="t('mes.labor.operatorCode')" prop="operatorCode" width="100" />
          <el-table-column :label="t('mes.labor.operatorName')" prop="operatorName" min-width="100" />
          <el-table-column :label="t('mes.labor.department')" prop="department" width="100" />
          <el-table-column :label="t('mes.labor.jobTitle')" prop="jobTitle" width="100" />
          <el-table-column :label="t('mes.labor.shiftGroup')" prop="shiftGroup" width="80" />
          <el-table-column :label="t('mes.labor.status')" prop="status" width="90">
            <template #default="{ row }">
              <el-tag :type="getOpStatusType(row.status)">
                {{ t(`mes.labor.operatorStatus${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('common.operations')" width="200" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click.stop="handleView(row)">{{ t('common.detail') }}</el-button>
              <el-button type="primary" link @click.stop="handleEdit(row)">{{ t('common.edit') }}</el-button>
              <el-button type="danger" link @click.stop="handleDelete(row)">{{ t('common.delete') }}</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane :label="t('mes.labor.attendance')" name="attendance">
        <div style="margin-bottom: 12px;">
          <el-button type="success" :icon="Top" @click="handleClockIn" style="margin-right: 8px;">
            {{ t('mes.labor.clockIn') }}
          </el-button>
          <el-button type="warning" :icon="Bottom" @click="handleClockOut">
            {{ t('mes.labor.clockOut') }}
          </el-button>
        </div>
        <el-table v-loading="attendanceLoading" :data="attendanceList" border>
          <el-table-column :label="t('mes.labor.operatorCode')" prop="operatorCode" width="100" />
          <el-table-column :label="t('mes.labor.operatorName')" prop="operatorName" min-width="100" />
          <el-table-column :label="t('mes.labor.date')" prop="attendanceDate" width="100" />
          <el-table-column :label="t('mes.labor.clockIn')" prop="clockIn" width="80" />
          <el-table-column :label="t('mes.labor.clockOut')" prop="clockOut" width="80" />
          <el-table-column :label="t('mes.labor.attendanceStatus')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="getAttStatusType(row.status)">
                {{ t(`mes.labor.attendanceStatus${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.labor.overtime')" width="70">
            <template #default="{ row }">
              {{ row.overtime ? t('common.yes') : t('common.no') }}
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane :label="t('mes.labor.timeRecords')" name="timeRecords">
        <div style="margin-bottom: 12px;">
          <el-input v-model="timeRecordQuery.operatorId" :placeholder="t('mes.labor.operatorIdPlaceholder')"
            class="filter-item" style="width: 160px; margin-right: 8px;" />
          <el-select v-model="timeRecordQuery.status" :placeholder="t('mes.labor.statusPlaceholder')"
            clearable style="width: 140px; margin-right: 8px;">
            <el-option v-for="s in recordStatusOptions" :key="s.value" :label="t(s.label)" :value="s.value" />
          </el-select>
          <el-button type="primary" :icon="Search" @click="loadTimeRecords">{{ t('common.query') }}</el-button>
        </div>
        <el-table v-loading="timeRecordLoading" :data="timeRecordList" border>
          <el-table-column :label="t('mes.labor.operatorCode')" prop="operatorCode" width="100" />
          <el-table-column :label="t('mes.labor.operatorName')" prop="operatorName" min-width="100" />
          <el-table-column :label="t('mes.labor.date')" prop="recordDate" width="100" />
          <el-table-column :label="t('mes.labor.startTime')" prop="startTime" width="130" />
          <el-table-column :label="t('mes.labor.endTime')" prop="endTime" width="130">
            <template #default="{ row }">{{ row.endTime ? formatDT(row.endTime) : '-' }}</template>
          </el-table-column>
          <el-table-column :label="t('mes.labor.totalHours')" prop="totalHours" width="80" />
          <el-table-column :label="t('mes.labor.recordType')" prop="recordType" width="80">
            <template #default="{ row }">{{ t(`mes.labor.recordType${row.recordType}`) }}</template>
          </el-table-column>
          <el-table-column :label="t('mes.labor.recordStatus')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="getRecStatusType(row.status)">
                {{ t(`mes.labor.recordStatus${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('common.operations')" width="200" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.status === 'DRAFT' && !row.endTime" type="primary" link @click="handleEndRecord(row)">
                {{ t('mes.labor.clockOut') }}
              </el-button>
              <el-button v-if="row.status === 'DRAFT' && row.endTime" type="primary" link @click="handleSubmitRecord(row)">
                {{ t('mes.labor.submit') }}
              </el-button>
              <el-button v-if="row.status === 'SUBMITTED'" type="success" link @click="handleApproveRecord(row)">
                {{ t('mes.labor.approve') }}
              </el-button>
              <el-button v-if="row.status === 'SUBMITTED'" type="danger" link @click="handleRejectRecord(row)">
                {{ t('mes.labor.reject') }}
              </el-button>
              <el-button type="danger" link @click="handleDeleteRecord(row)">{{ t('common.delete') }}</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane :label="t('mes.labor.assignments')" name="assignments">
        <div style="margin-bottom: 12px;">
          <el-button type="primary" :icon="Plus" @click="showAssignDialog = true">
            {{ t('mes.labor.newAssignment') }}
          </el-button>
        </div>
        <el-table v-loading="assignmentLoading" :data="assignmentList" border>
          <el-table-column :label="t('mes.labor.operatorId')" prop="operatorId" width="80" />
          <el-table-column :label="t('mes.labor.workCenterId')" prop="workCenterId" width="120" />
          <el-table-column :label="t('mes.labor.workCenterName')" prop="workCenterName" min-width="120" />
          <el-table-column :label="t('mes.labor.shiftType')" prop="shiftType" width="80">
            <template #default="{ row }">{{ t(`mes.labor.shiftType${row.shiftType}`) }}</template>
          </el-table-column>
          <el-table-column :label="t('mes.labor.assignmentStatus')" prop="status" width="90">
            <template #default="{ row }">
              <el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'">
                {{ t(`mes.labor.assignmentStatus${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('common.operations')" width="120">
            <template #default="{ row }">
              <el-button v-if="row.status === 'ACTIVE'" type="danger" link @click="handleEndAssignment(row)">
                {{ t('mes.labor.endAssignment') }}
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <el-dialog :title="t('mes.labor.newAssignment')" v-model="showAssignDialog" width="500px">
      <el-form :model="assignForm" label-width="120px">
        <el-form-item :label="t('mes.labor.operatorId')" :required="true">
          <el-input-number v-model="assignForm.operatorId" :min="1" style="width: 100%" />
        </el-form-item>
        <el-form-item :label="t('mes.labor.workCenterId')" :required="true">
          <el-input v-model="assignForm.workCenterId" />
        </el-form-item>
        <el-form-item :label="t('mes.labor.workCenterName')">
          <el-input v-model="assignForm.workCenterName" />
        </el-form-item>
        <el-form-item :label="t('mes.labor.shiftType')">
          <el-select v-model="assignForm.shiftType" style="width: 100%">
            <el-option label="DAY" value="DAY" />
            <el-option label="NIGHT" value="NIGHT" />
            <el-option label="MIDDLE" value="MIDDLE" />
            <el-option label="ROTATING" value="ROTATING" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAssignDialog = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="assignSubmitting" @click="submitAssignment">
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
  getOperatorListAPI, deleteOperatorAPI,
  clockInAPI, clockOutAPI, getAttendanceListAPI,
  startTimeRecordAPI, endTimeRecordAPI, submitTimeRecordAPI,
  approveTimeRecordAPI, rejectTimeRecordAPI, getTimeRecordListAPI,
  deleteTimeRecordAPI,
  createAssignmentAPI, endAssignmentAPI, getAssignmentListAPI,
  type Operator, type Attendance, type TimeRecord, type WorkCenterAssignment,
} from '@/apis/labor'

const { t } = useI18n()
const router = useRouter()
const activeTab = ref('operators')

const operatorLoading = ref(false)
const attendanceLoading = ref(false)
const timeRecordLoading = ref(false)
const assignmentLoading = ref(false)

const operatorList = ref<Operator[]>([])
const attendanceList = ref<Attendance[]>([])
const timeRecordList = ref<TimeRecord[]>([])
const assignmentList = ref<WorkCenterAssignment[]>([])

const operatorQuery = ref({ operatorCode: '', department: '', status: '' })
const timeRecordQuery = ref({ operatorId: '', status: '' })
const showAssignDialog = ref(false)
const assignSubmitting = ref(false)
const assignForm = ref({ operatorId: 1, workCenterId: '', workCenterName: '', shiftType: 'DAY' as const })

const operatorStatusOptions = [
  { value: 'ACTIVE', label: 'mes.labor.operatorStatusACTIVE' },
  { value: 'INACTIVE', label: 'mes.labor.operatorStatusINACTIVE' },
  { value: 'ON_LEAVE', label: 'mes.labor.operatorStatusON_LEAVE' },
  { value: 'TERMINATED', label: 'mes.labor.operatorStatusTERMINATED' },
]
const recordStatusOptions = [
  { value: 'DRAFT', label: 'mes.labor.recordStatusDRAFT' },
  { value: 'SUBMITTED', label: 'mes.labor.recordStatusSUBMITTED' },
  { value: 'APPROVED', label: 'mes.labor.recordStatusAPPROVED' },
  { value: 'REJECTED', label: 'mes.labor.recordStatusREJECTED' },
]

function getOpStatusType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = { ACTIVE: 'success', INACTIVE: 'info', ON_LEAVE: 'warning', TERMINATED: 'danger' }
  return map[s] || 'info'
}
function getSkillLevelType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = { TRAINEE: 'info', JUNIOR: 'primary', MIDDLE: 'warning', SENIOR: 'danger', MASTER: 'success' }
  return map[s] || 'info'
}
function getAttStatusType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = { PRESENT: 'success', LATE: 'warning', ABSENT: 'danger', HALF_DAY: 'info', OVERTIME: 'primary', VACATION: 'info', SICK: 'warning', BUSINESS_TRIP: 'primary' }
  return map[s] || 'info'
}
function getRecStatusType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = { DRAFT: 'info', SUBMITTED: 'primary', APPROVED: 'success', REJECTED: 'danger' }
  return map[s] || 'info'
}
function formatDT(dt: string) { return dt ? new Date(dt).toLocaleString() : '-' }

async function loadOperators() {
  operatorLoading.value = true
  try {
    const params: Record<string, unknown> = {}
    if (operatorQuery.value.operatorCode) params.operatorCode = operatorQuery.value.operatorCode
    if (operatorQuery.value.department) params.department = operatorQuery.value.department
    if (operatorQuery.value.status) params.status = operatorQuery.value.status
    const res = await getOperatorListAPI(params)
    operatorList.value = res.data || []
  } catch { ElMessage.error(t('mes.labor.loadFailed'))
  } finally { operatorLoading.value = false }
}
async function loadAttendance() {
  attendanceLoading.value = true
  try {
    const res = await getAttendanceListAPI()
    attendanceList.value = res.data || []
  } catch { /* silent */
  } finally { attendanceLoading.value = false }
}
async function loadTimeRecords() {
  timeRecordLoading.value = true
  try {
    const params: Record<string, unknown> = {}
    if (timeRecordQuery.value.operatorId) params.operatorId = Number(timeRecordQuery.value.operatorId)
    if (timeRecordQuery.value.status) params.status = timeRecordQuery.value.status
    const res = await getTimeRecordListAPI(params)
    timeRecordList.value = res.data || []
  } catch { /* silent */
  } finally { timeRecordLoading.value = false }
}
async function loadAssignments() {
  assignmentLoading.value = true
  try {
    const res = await getAssignmentListAPI()
    assignmentList.value = res.data || []
  } catch { /* silent */
  } finally { assignmentLoading.value = false }
}

function handleFilter() { loadOperators() }
function handleCreate() { router.push({ name: 'laborForm' }) }
function handleView(row: Operator) { router.push({ name: 'laborDetail', query: { id: row.id } }) }
function handleEdit(row: Operator) { router.push({ name: 'laborForm', query: { id: row.id } }) }
async function handleDelete(row: Operator) {
  try {
    await ElMessageBox.confirm(t('mes.labor.confirmDelete'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await deleteOperatorAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    loadOperators()
  } catch { /* cancel */ }
}

async function handleClockIn() {
  try {
    const { value: oid } = await ElMessageBox.prompt(t('mes.labor.operatorIdPrompt'), t('mes.labor.clockIn'),
      { inputPattern: /^\d+$/, inputErrorMessage: t('mes.labor.operatorIdInvalid') })
    await clockInAPI(Number(oid))
    ElMessage.success(t('mes.labor.clockInSuccess'))
    loadAttendance()
  } catch { /* cancel */ }
}
async function handleClockOut() {
  try {
    const { value: oid } = await ElMessageBox.prompt(t('mes.labor.operatorIdPrompt'), t('mes.labor.clockOut'),
      { inputPattern: /^\d+$/, inputErrorMessage: t('mes.labor.operatorIdInvalid') })
    await clockOutAPI(Number(oid))
    ElMessage.success(t('mes.labor.clockOutSuccess'))
    loadAttendance()
  } catch { /* cancel */ }
}

async function handleEndRecord(row: TimeRecord) {
  try { await endTimeRecordAPI(row.id!); ElMessage.success(t('mes.labor.endSuccess')); loadTimeRecords() }
  catch { /* error */ }
}
async function handleSubmitRecord(row: TimeRecord) {
  try { await submitTimeRecordAPI(row.id!); ElMessage.success(t('mes.labor.submitSuccess')); loadTimeRecords() }
  catch { /* error */ }
}
async function handleApproveRecord(row: TimeRecord) {
  try {
    const { value } = await ElMessageBox.prompt(t('mes.labor.approvedByPrompt'), t('mes.labor.approve'),
      { inputPattern: /.+/, inputErrorMessage: t('mes.labor.approvedByRequired') })
    await approveTimeRecordAPI(row.id!, value)
    ElMessage.success(t('mes.labor.approveSuccess'))
    loadTimeRecords()
  } catch { /* cancel */ }
}
async function handleRejectRecord(row: TimeRecord) {
  try {
    const { value } = await ElMessageBox.prompt(t('mes.labor.approvedByPrompt'), t('mes.labor.reject'),
      { inputPattern: /.+/, inputErrorMessage: t('mes.labor.approvedByRequired') })
    await rejectTimeRecordAPI(row.id!, value)
    ElMessage.success(t('mes.labor.rejectSuccess'))
    loadTimeRecords()
  } catch { /* cancel */ }
}
async function handleDeleteRecord(row: TimeRecord) {
  try {
    await ElMessageBox.confirm(t('mes.labor.confirmDeleteRecord'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await deleteTimeRecordAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    loadTimeRecords()
  } catch { /* cancel */ }
}

async function handleEndAssignment(row: WorkCenterAssignment) {
  try {
    await ElMessageBox.confirm(t('mes.labor.confirmEndAssignment'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await endAssignmentAPI(row.id!)
    ElMessage.success(t('mes.labor.endAssignmentSuccess'))
    loadAssignments()
  } catch { /* cancel */ }
}

async function submitAssignment() {
  assignSubmitting.value = true
  try {
    await createAssignmentAPI(assignForm.value)
    ElMessage.success(t('mes.labor.createAssignmentSuccess'))
    showAssignDialog.value = false
    loadAssignments()
  } catch { ElMessage.error(t('common.submitFailed'))
  } finally { assignSubmitting.value = false }
}

onMounted(() => {
  loadOperators(); loadAttendance(); loadTimeRecords(); loadAssignments()
})
</script>
