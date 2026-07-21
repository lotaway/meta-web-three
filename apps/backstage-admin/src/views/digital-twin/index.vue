<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import {
  getDevicesAPI, registerDeviceAPI, updateDeviceStatusAPI, updateDevicePositionAPI,
  getWorkshopsAPI, createWorkshopAPI,
  getProductionLinesAPI, createProductionLineAPI,
  getActiveAlertsAPI, acknowledgeAlertAPI, resolveAlertAPI, createAlertAPI,
  getStatsSummaryAPI,
  getAlertRulesAPI, createAlertRuleAPI, updateAlertRuleAPI,
  enableAlertRuleAPI, disableAlertRuleAPI, deleteAlertRuleAPI,
  type DigitalTwinDevice, type StatsSummary,
  type Workshop, type ProductionLine, type Alert, type AlertRule,
} from '@/apis/digitalTwin'

const activeTab = ref('dashboard')
const loading = ref(false)

// Dashboard
const stats = ref<StatsSummary>({ onlineDeviceCount: 0, activeAlertCount: 0, averageEfficiency: 0 })
const loadStats = async () => {
  try { const res = await getStatsSummaryAPI(); stats.value = res.data } catch (_) { /* ignore */ }
}

// Devices
const devices = ref<DigitalTwinDevice[]>([])
const loadDevices = async () => {
  try { const res = await getDevicesAPI(); devices.value = res.data as any || [] } catch (_) { /* ignore */ }
}
const deviceDialog = ref(false)
const deviceForm = ref({ deviceCode: '', deviceName: '', deviceType: 'CNC', workshopId: '', productionLineId: '' })

const createDevice = async () => {
  try {
    await registerDeviceAPI(deviceForm.value)
    ElMessage.success('Device registered')
    deviceDialog.value = false
    loadDevices()
  } catch (_) { ElMessage.error('Failed to register device') }
}

const updateStatus = async (code: string, status: string) => {
  try {
    await updateDeviceStatusAPI(code, status)
    ElMessage.success('Status updated')
    loadDevices()
  } catch (_) { ElMessage.error('Failed to update status') }
}

// Workshops
const workshops = ref<Workshop[]>([])
const loadWorkshops = async () => {
  try { const res = await getWorkshopsAPI(); workshops.value = res.data as any || [] } catch (_) { /* ignore */ }
}
const workshopDialog = ref(false)
const workshopForm = ref({ workshopCode: '', workshopName: '', description: '' })
const createWs = async () => {
  try {
    await createWorkshopAPI(workshopForm.value)
    ElMessage.success('Workshop created')
    workshopDialog.value = false
    loadWorkshops()
  } catch (_) { ElMessage.error('Failed to create workshop') }
}

// Production Lines
const lines = ref<ProductionLine[]>([])
const loadLines = async () => {
  try { const res = await getProductionLinesAPI(); lines.value = res.data as any || [] } catch (_) { /* ignore */ }
}
const lineDialog = ref(false)
const lineForm = ref({ lineCode: '', lineName: '', workshopId: '', capacity: 10 })
const createPl = async () => {
  try {
    await createProductionLineAPI(lineForm.value)
    ElMessage.success('Production line created')
    lineDialog.value = false
    loadLines()
  } catch (_) { ElMessage.error('Failed to create production line') }
}

// Active Alerts
const alerts = ref<Alert[]>([])
const loadAlerts = async () => {
  try { const res = await getActiveAlertsAPI(); alerts.value = res.data as any || [] } catch (_) { /* ignore */ }
}
const alertDialog = ref(false)
const alertForm = ref({ deviceCode: '', workshopId: '', level: 'WARNING', type: 'PRODUCTION', title: '', description: '' })
const createAlert = async () => {
  try {
    await createAlertAPI(alertForm.value)
    ElMessage.success('Alert created')
    alertDialog.value = false
    loadAlerts()
  } catch (_) { ElMessage.error('Failed to create alert') }
}
const ackAlert = async (id: number) => {
  try {
    await ElMessageBox.prompt('Enter your name:', 'Acknowledge').then(async ({ value }) => {
      if (!value) return
      await acknowledgeAlertAPI(id, value)
      ElMessage.success('Alert acknowledged')
      loadAlerts()
    })
  } catch (_) { /* ignore */ }
}
const resolveAlert = async (id: number) => {
  try {
    await ElMessageBox.prompt('Enter solution:', 'Resolve').then(async ({ value: solution }) => {
      if (!solution) return
      await ElMessageBox.prompt('Enter resolver name:', 'Resolve').then(async ({ value: resolvedBy }) => {
        if (!resolvedBy) return
        await resolveAlertAPI(id, solution, resolvedBy)
        ElMessage.success('Alert resolved')
        loadAlerts()
      })
    })
  } catch (_) { /* ignore */ }
}

// Alert Rules
const rules = ref<AlertRule[]>([])
const loadRules = async () => {
  try { const res = await getAlertRulesAPI(); rules.value = res.data || [] } catch (_) { /* ignore */ }
}
const ruleDialog = ref(false)
const ruleForm = ref<Record<string, any>>({})
const editRuleId = ref<number | undefined>(undefined)

const openRuleForm = async (rule?: AlertRule) => {
  editRuleId.value = rule?.id
  ruleForm.value = rule ? { ...rule } : {
    ruleCode: '', ruleName: '', description: '', deviceType: '', deviceCode: '',
    workshopId: '', metricType: 'TEMPERATURE', operator: 'GT', thresholdValue: 80,
    durationSeconds: 60, level: 'WARNING', alertType: 'PRODUCTION',
    titleTemplate: '', descriptionTemplate: '', enabled: true,
  }
  ruleDialog.value = true
}

const saveRule = async () => {
  try {
    if (editRuleId.value) {
      await updateAlertRuleAPI(editRuleId.value, ruleForm.value)
    } else {
      await createAlertRuleAPI(ruleForm.value)
    }
    ElMessage.success('Rule saved')
    ruleDialog.value = false
    loadRules()
  } catch (_) { ElMessage.error('Failed to save rule') }
}

const toggleRule = async (id: number, enable: boolean) => {
  try {
    if (enable) await enableAlertRuleAPI(id)
    else await disableAlertRuleAPI(id)
    ElMessage.success(enable ? 'Rule enabled' : 'Rule disabled')
    loadRules()
  } catch (_) { ElMessage.error('Failed to toggle rule') }
}

const removeRule = async (id: number) => {
  try {
    await ElMessageBox.confirm('Delete this rule?', 'Confirm', { type: 'warning' })
    await deleteAlertRuleAPI(id)
    ElMessage.success('Rule deleted')
    loadRules()
  } catch (_) { /* ignore */ }
}

const refreshAll = () => {
  loadStats(); loadDevices(); loadWorkshops(); loadLines(); loadAlerts(); loadRules()
}

onMounted(refreshAll)

const statusTag = (s?: string): 'success' | 'warning' | 'danger' | 'info' => {
  const m: Record<string, any> = { RUNNING: 'success', IDLE: 'info', MAINTENANCE: 'warning', ERROR: 'danger', OFFLINE: 'danger' }
  return m[s || ''] || 'info'
}
const levelTag = (l?: string): 'success' | 'warning' | 'danger' | 'info' => {
  const m: Record<string, any> = { INFO: 'info', WARNING: 'warning', ERROR: 'danger', CRITICAL: 'danger' }
  return m[l || ''] || 'info'
}
</script>

<template>
  <div class="dt-container">
    <div class="toolbar">
      <el-button :icon="Refresh" @click="refreshAll">Refresh</el-button>
    </div>

    <!-- Dashboard -->
    <el-row :gutter="20" class="stat-row" v-if="activeTab === 'dashboard'">
      <el-col :span="8"><el-card shadow="hover"><div class="stat-value">{{ stats.onlineDeviceCount }}</div><div class="stat-label">Online Devices</div></el-card></el-col>
      <el-col :span="8"><el-card shadow="hover"><div class="stat-value">{{ stats.activeAlertCount }}</div><div class="stat-label">Active Alerts</div></el-card></el-col>
      <el-col :span="8"><el-card shadow="hover"><div class="stat-value">{{ (stats.averageEfficiency * 100).toFixed(1) }}%</div><div class="stat-label">Avg Efficiency</div></el-card></el-col>
    </el-row>

    <el-tabs v-model="activeTab" type="border-card">
      <el-tab-pane label="Devices" name="devices">
        <div class="section-toolbar">
          <el-button type="primary" @click="deviceDialog = true">Register Device</el-button>
        </div>
        <el-table :data="devices" border stripe v-loading="loading">
          <el-table-column prop="deviceCode" label="Code" width="130" />
          <el-table-column prop="deviceName" label="Name" min-width="140" />
          <el-table-column prop="deviceType" label="Type" width="100" />
          <el-table-column prop="workshopId" label="Workshop" width="100" />
          <el-table-column prop="efficiency" label="Efficiency" width="90">
            <template #default="{ row }">{{ (row.efficiency * 100).toFixed(1) }}%</template>
          </el-table-column>
          <el-table-column prop="status" label="Status" width="100">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="200" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.status === 'RUNNING'" link type="warning" size="small" @click="updateStatus(row.deviceCode, 'IDLE')">Idle</el-button>
              <el-button v-if="row.status === 'IDLE'" link type="success" size="small" @click="updateStatus(row.deviceCode, 'RUNNING')">Run</el-button>
              <el-button link type="danger" size="small" @click="updateStatus(row.deviceCode, 'ERROR')">Error</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="Workshops" name="workshops">
        <div class="section-toolbar">
          <el-button type="primary" @click="workshopDialog = true">Create Workshop</el-button>
        </div>
        <el-table :data="workshops" border stripe>
          <el-table-column prop="workshopCode" label="Code" width="130" />
          <el-table-column prop="workshopName" label="Name" min-width="200" />
          <el-table-column prop="description" label="Description" min-width="250" />
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="Production Lines" name="lines">
        <div class="section-toolbar">
          <el-button type="primary" @click="lineDialog = true">Create Line</el-button>
        </div>
        <el-table :data="lines" border stripe>
          <el-table-column prop="lineCode" label="Code" width="130" />
          <el-table-column prop="lineName" label="Name" min-width="200" />
          <el-table-column prop="workshopId" label="Workshop" width="120" />
          <el-table-column prop="capacity" label="Capacity" width="90" />
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="Active Alerts" name="alerts">
        <div class="section-toolbar">
          <el-button type="primary" @click="alertDialog = true">Create Alert</el-button>
        </div>
        <el-table :data="alerts" border stripe>
          <el-table-column prop="alertCode" label="Code" width="130" />
          <el-table-column prop="deviceCode" label="Device" width="120" />
          <el-table-column prop="title" label="Title" min-width="160" />
          <el-table-column prop="level" label="Level" width="90">
            <template #default="{ row }"><el-tag :type="levelTag(row.level)">{{ row.level }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="status" label="Status" width="100">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="occurredAt" label="Occurred" width="170">
            <template #default="{ row }">{{ row.occurredAt || '-' }}</template>
          </el-table-column>
          <el-table-column label="Actions" width="200" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.status === 'OPEN'" link type="primary" size="small" @click="ackAlert(row.id)">Ack</el-button>
              <el-button v-if="row.status !== 'RESOLVED'" link type="success" size="small" @click="resolveAlert(row.id)">Resolve</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="Alert Rules" name="rules">
        <div class="section-toolbar">
          <el-button type="primary" @click="openRuleForm()">Create Rule</el-button>
        </div>
        <el-table :data="rules" border stripe>
          <el-table-column prop="ruleCode" label="Code" width="110" />
          <el-table-column prop="ruleName" label="Name" min-width="150" />
          <el-table-column prop="deviceType" label="Device Type" width="110" />
          <el-table-column prop="metricType" label="Metric" width="110" />
          <el-table-column prop="operator" label="Op" width="60" />
          <el-table-column prop="thresholdValue" label="Threshold" width="90" />
          <el-table-column prop="level" label="Level" width="80">
            <template #default="{ row }"><el-tag :type="levelTag(row.level)" size="small">{{ row.level }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="enabled" label="Enabled" width="80">
            <template #default="{ row }"><el-tag :type="row.enabled ? 'success' : 'info'" size="small">{{ row.enabled ? 'Yes' : 'No' }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="220" fixed="right">
            <template #default="{ row }">
              <el-button link type="primary" size="small" @click="openRuleForm(row)">Edit</el-button>
              <el-button v-if="row.enabled" link type="warning" size="small" @click="toggleRule(row.id, false)">Disable</el-button>
              <el-button v-else link type="success" size="small" @click="toggleRule(row.id, true)">Enable</el-button>
              <el-button link type="danger" size="small" @click="removeRule(row.id)">Delete</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <!-- Dialogs -->
    <el-dialog v-model="deviceDialog" title="Register Device" width="450px">
      <el-form :model="deviceForm" label-width="140px">
        <el-form-item label="Device Code"><el-input v-model="deviceForm.deviceCode" /></el-form-item>
        <el-form-item label="Device Name"><el-input v-model="deviceForm.deviceName" /></el-form-item>
        <el-form-item label="Device Type">
          <el-select v-model="deviceForm.deviceType" style="width:100%">
            <el-option label="CNC" value="CNC" /><el-option label="Robot" value="ROBOT" />
            <el-option label="Conveyor" value="CONVEYOR" /><el-option label="Sensor" value="SENSOR" />
            <el-option label="Printer" value="PRINTER" /><el-option label="Other" value="OTHER" />
          </el-select>
        </el-form-item>
        <el-form-item label="Workshop ID"><el-input v-model="deviceForm.workshopId" /></el-form-item>
        <el-form-item label="Production Line ID"><el-input v-model="deviceForm.productionLineId" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="deviceDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createDevice">Register</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="workshopDialog" title="Create Workshop" width="450px">
      <el-form :model="workshopForm" label-width="140px">
        <el-form-item label="Workshop Code"><el-input v-model="workshopForm.workshopCode" /></el-form-item>
        <el-form-item label="Workshop Name"><el-input v-model="workshopForm.workshopName" /></el-form-item>
        <el-form-item label="Description"><el-input v-model="workshopForm.description" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="workshopDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createWs">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="lineDialog" title="Create Production Line" width="450px">
      <el-form :model="lineForm" label-width="140px">
        <el-form-item label="Line Code"><el-input v-model="lineForm.lineCode" /></el-form-item>
        <el-form-item label="Line Name"><el-input v-model="lineForm.lineName" /></el-form-item>
        <el-form-item label="Workshop ID"><el-input v-model="lineForm.workshopId" /></el-form-item>
        <el-form-item label="Capacity"><el-input-number v-model="lineForm.capacity" :min="1" style="width:100%" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="lineDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createPl">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="alertDialog" title="Create Alert" width="450px">
      <el-form :model="alertForm" label-width="140px">
        <el-form-item label="Device Code"><el-input v-model="alertForm.deviceCode" /></el-form-item>
        <el-form-item label="Workshop ID"><el-input v-model="alertForm.workshopId" /></el-form-item>
        <el-form-item label="Level">
          <el-select v-model="alertForm.level" style="width:100%">
            <el-option label="Info" value="INFO" /><el-option label="Warning" value="WARNING" />
            <el-option label="Error" value="ERROR" /><el-option label="Critical" value="CRITICAL" />
          </el-select>
        </el-form-item>
        <el-form-item label="Type">
          <el-select v-model="alertForm.type" style="width:100%">
            <el-option label="Production" value="PRODUCTION" /><el-option label="Quality" value="QUALITY" />
            <el-option label="Equipment" value="EQUIPMENT" /><el-option label="Safety" value="SAFETY" />
          </el-select>
        </el-form-item>
        <el-form-item label="Title"><el-input v-model="alertForm.title" /></el-form-item>
        <el-form-item label="Description"><el-input v-model="alertForm.description" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="alertDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createAlert">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="ruleDialog" :title="editRuleId ? 'Edit Rule' : 'Create Rule'" width="550px">
      <el-form :model="ruleForm" label-width="140px">
        <el-row :gutter="16">
          <el-col :span="12"><el-form-item label="Rule Code"><el-input v-model="ruleForm.ruleCode" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="Rule Name"><el-input v-model="ruleForm.ruleName" /></el-form-item></el-col>
        </el-row>
        <el-form-item label="Description"><el-input v-model="ruleForm.description" /></el-form-item>
        <el-row :gutter="16">
          <el-col :span="12"><el-form-item label="Device Type"><el-input v-model="ruleForm.deviceType" /></el-form-item></el-col>
          <el-col :span="12"><el-form-item label="Metric Type">
            <el-select v-model="ruleForm.metricType" style="width:100%">
              <el-option label="Temperature" value="TEMPERATURE" /><el-option label="Pressure" value="PRESSURE" />
              <el-option label="Speed" value="SPEED" /><el-option label="Vibration" value="VIBRATION" />
            </el-select>
          </el-form-item></el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="8"><el-form-item label="Operator">
            <el-select v-model="ruleForm.operator" style="width:100%">
              <el-option label=">" value="GT" /><el-option label=">=" value="GE" />
              <el-option label="=" value="EQ" /><el-option label="<" value="LT" />
              <el-option label="<=" value="LE" />
            </el-select>
          </el-form-item></el-col>
          <el-col :span="8"><el-form-item label="Threshold"><el-input-number v-model="ruleForm.thresholdValue" style="width:100%" /></el-form-item></el-col>
          <el-col :span="8"><el-form-item label="Duration(s)"><el-input-number v-model="ruleForm.durationSeconds" :min="0" style="width:100%" /></el-form-item></el-col>
        </el-row>
        <el-row :gutter="16">
          <el-col :span="12"><el-form-item label="Level">
            <el-select v-model="ruleForm.level" style="width:100%">
              <el-option label="Info" value="INFO" /><el-option label="Warning" value="WARNING" />
              <el-option label="Error" value="ERROR" /><el-option label="Critical" value="CRITICAL" />
            </el-select>
          </el-form-item></el-col>
          <el-col :span="12"><el-form-item label="Alert Type">
            <el-select v-model="ruleForm.alertType" style="width:100%">
              <el-option label="Production" value="PRODUCTION" /><el-option label="Quality" value="QUALITY" />
              <el-option label="Equipment" value="EQUIPMENT" /><el-option label="Safety" value="SAFETY" />
            </el-select>
          </el-form-item></el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="ruleDialog = false">Cancel</el-button>
        <el-button type="primary" @click="saveRule">Save</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.dt-container { padding: 20px; }
.toolbar { margin-bottom: 16px; }
.stat-row { margin-bottom: 20px; }
.stat-value { font-size: 28px; font-weight: bold; color: #303133; text-align: center; }
.stat-label { font-size: 14px; color: #909399; text-align: center; margin-top: 5px; }
.section-toolbar { margin-bottom: 16px; }
</style>
