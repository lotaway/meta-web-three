<template>
  <div class="app-container">
    <div class="filter-container">
      <el-select v-model="selectedEquipment" filterable :placeholder="t('mes.scada.selectEquipment')"
        style="width: 220px" class="filter-item" @change="handleEquipmentChange">
        <el-option v-for="eq in equipmentList" :key="eq.equipmentCode"
          :label="eq.equipmentName + ' (' + eq.equipmentCode + ')'" :value="eq.equipmentCode" />
      </el-select>
      <el-button type="primary" :icon="Refresh" class="filter-item" @click="refreshData">
        {{ t('common.refresh') }}
      </el-button>
      <el-tag v-if="selectedEquipment && latestTelemetry.length > 0" type="success" effect="dark" class="filter-item">
        {{ t('mes.scada.lastUpdate') }}: {{ latestTelemetry[0]?.collectTime ? formatTime(latestTelemetry[0].collectTime) : '-' }}
      </el-tag>
    </div>

    <el-row :gutter="16">
      <!-- Real-time Metrics -->
      <el-col :span="16">
        <el-card v-loading="metricsLoading" shadow="hover">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><Monitor /></el-icon>
              {{ t('mes.scada.realtimeMetrics') }}
            </span>
          </template>
          <div v-if="latestMetrics.length > 0">
            <el-row :gutter="12">
              <el-col :span="8" v-for="m in latestMetrics" :key="m.metricCode" style="margin-bottom: 12px;">
                <el-card shadow="never" class="metric-card" :class="{ 'metric-alert': isOutOfRange(m) }">
                  <div class="metric-name">{{ m.metricName || m.metricCode }}</div>
                  <div class="metric-value" :style="{ color: isOutOfRange(m) ? '#f56c6c' : '#303133' }">
                    {{ m.value }} <span class="metric-unit">{{ m.unit || '' }}</span>
                  </div>
                  <div v-if="m.upperLimit != null" class="metric-range">
                    {{ m.lowerLimit != null ? m.lowerLimit : '-' }} ~ {{ m.upperLimit }}
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
          <el-empty v-else-if="selectedEquipment" :description="t('mes.scada.noTelemetry')" />
          <el-empty v-else :description="t('mes.scada.selectEquipmentHint')" />
        </el-card>

        <!-- Telemetry History -->
        <el-card shadow="hover" style="margin-top: 16px;">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><DataLine /></el-icon>
              {{ t('mes.scada.telemetryHistory') }}
            </span>
          </template>
          <el-table v-loading="historyLoading" :data="telemetryHistory" size="small" border style="width: 100%">
            <el-table-column type="expand">
              <template #default="{ row }">
                <div v-if="row.metrics && row.metrics.length > 0" style="padding: 8px;">
                  <el-descriptions :column="3" size="small" border>
                    <el-descriptions-item v-for="m in row.metrics" :key="m.metricCode"
                      :label="m.metricName || m.metricCode">
                      {{ m.value }} {{ m.unit || '' }}
                      <el-tag v-if="isOutOfRange(m)" type="danger" size="small" style="margin-left: 4px;">!</el-tag>
                    </el-descriptions-item>
                  </el-descriptions>
                </div>
              </template>
            </el-table-column>
            <el-table-column :label="t('common.id')" prop="id" width="50" />
            <el-table-column :label="t('mes.scada.collectTime')" prop="collectTime" width="150">
              <template #default="{ row }">
                {{ row.collectTime ? formatTime(row.collectTime) : '-' }}
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.scada.metrics')" min-width="200">
              <template #default="{ row }">
                <span v-for="(m, i) in row.metrics?.slice(0, 5)" :key="m.metricCode">
                  <el-tag :type="isOutOfRange(m) ? 'danger' : 'info'" size="small" style="margin-right: 4px; margin-bottom: 4px;">
                    {{ m.metricName || m.metricCode }}: {{ m.value }}{{ m.unit || '' }}
                  </el-tag>
                </span>
                <span v-if="row.metrics && row.metrics.length > 5" style="color: #999;">
                  +{{ row.metrics.length - 5 }}
                </span>
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.scada.topic')" prop="topic" width="150" />
          </el-table>
        </el-card>
      </el-col>

      <!-- Command Panel -->
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><Connection /></el-icon>
              {{ t('mes.scada.deviceCommand') }}
            </span>
          </template>
          <el-form :model="commandForm" label-width="80px" size="small">
            <el-form-item :label="t('mes.scada.commandType')">
              <el-select v-model="commandForm.commandType" style="width: 100%">
                <el-option v-for="ct in commandTypeOptions" :key="ct.value" :label="ct.label" :value="ct.value" />
              </el-select>
            </el-form-item>
            <el-form-item :label="t('mes.scada.payload')">
              <el-input v-model="commandForm.payload" type="textarea" :rows="3"
                :placeholder="t('mes.scada.payloadPlaceholder')" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="cmdSending" @click="sendCommand" style="width: 100%">
                {{ t('mes.scada.sendCommand') }}
              </el-button>
            </el-form-item>
          </el-form>

          <el-divider />

          <div class="command-list-header">
            <strong>{{ t('mes.scada.recentCommands') }}</strong>
          </div>
          <div v-loading="cmdLoading">
            <div v-for="cmd in recentCommands" :key="cmd.id" class="command-item">
              <div class="command-header">
                <el-tag :type="getCmdStatusType(cmd.status)" size="small">
                  {{ cmd.commandType }}
                </el-tag>
                <el-tag :type="getCmdStatusType(cmd.status)" size="small" style="margin-left: 4px;">
                  {{ t(`mes.scada.cmdStatus${cmd.status}`) }}
                </el-tag>
              </div>
              <div class="command-payload">{{ cmd.payload || '-' }}</div>
              <div class="command-time">{{ cmd.createdAt ? formatTime(cmd.createdAt) : '-' }}</div>
            </div>
            <el-empty v-if="!selectedEquipment && recentCommands.length === 0"
              :description="t('mes.scada.selectEquipmentHint')" />
            <el-empty v-else-if="recentCommands.length === 0" :description="t('mes.scada.noCommands')" />
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { Refresh, Monitor, DataLine, Connection } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  getTelemetryListAPI,
  dispatchCommandAPI,
  getCommandListAPI,
  type TelemetryRecord,
  type TelemetryMetric,
  type DeviceCommand,
  type CommandType,
} from '@/apis/scada'
import { getEquipmentListAPI, type Equipment } from '@/apis/equipment'

const { t } = useI18n()

const equipmentList = ref<Equipment[]>([])
const selectedEquipment = ref('')
const metricsLoading = ref(false)
const historyLoading = ref(false)

const latestTelemetry = ref<TelemetryRecord[]>([])
const latestMetrics = ref<TelemetryMetric[]>([])
const telemetryHistory = ref<TelemetryRecord[]>([])

const cmdSending = ref(false)
const cmdLoading = ref(false)
const recentCommands = ref<DeviceCommand[]>([])

const commandForm = ref({
  commandType: 'CUSTOM' as CommandType,
  payload: '',
})

const commandTypeOptions = [
  { value: 'START', label: t('mes.scada.cmdTypeSTART') },
  { value: 'STOP', label: t('mes.scada.cmdTypeSTOP') },
  { value: 'RESET', label: t('mes.scada.cmdTypeRESET') },
  { value: 'SET_PARAMETER', label: t('mes.scada.cmdTypeSET_PARAMETER') },
  { value: 'CALIBRATE', label: t('mes.scada.cmdTypeCALIBRATE') },
  { value: 'SET_SPEED', label: t('mes.scada.cmdTypeSET_SPEED') },
  { value: 'SET_TEMPERATURE', label: t('mes.scada.cmdTypeSET_TEMPERATURE') },
  { value: 'CUSTOM', label: t('mes.scada.cmdTypeCUSTOM') },
]

function isOutOfRange(m: TelemetryMetric): boolean {
  if (m.upperLimit != null && m.value > m.upperLimit) return true
  if (m.lowerLimit != null && m.value < m.lowerLimit) return true
  return false
}

function formatTime(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

function getCmdStatusType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'info', SENT: 'primary', DELIVERED: 'warning', EXECUTED: 'success', FAILED: 'danger', TIMEOUT: 'danger',
  }
  return map[s] || 'info'
}

async function loadEquipment() {
  try {
    const res = await getEquipmentListAPI()
    equipmentList.value = res.data || []
  } catch {
    // silent
  }
}

async function loadTelemetry() {
  if (!selectedEquipment.value) return
  metricsLoading.value = true
  historyLoading.value = true
  try {
    const res = await getTelemetryListAPI(selectedEquipment.value, 20)
    telemetryHistory.value = res.data || []
    const metricsMap = new Map<string, TelemetryMetric>()
    for (const record of telemetryHistory.value.slice(0, 5)) {
      if (record.metrics) {
        for (const m of record.metrics) {
          metricsMap.set(m.metricCode, m)
        }
      }
    }
    latestMetrics.value = Array.from(metricsMap.values())
    latestTelemetry.value = telemetryHistory.value.slice(0, 5)
  } catch {
    ElMessage.error(t('mes.scada.loadFailed'))
  } finally {
    metricsLoading.value = false
    historyLoading.value = false
  }
}

async function loadCommands() {
  if (!selectedEquipment.value) return
  cmdLoading.value = true
  try {
    const res = await getCommandListAPI(selectedEquipment.value)
    recentCommands.value = (res.data || []).slice(0, 10)
  } catch {
    // silent
  } finally {
    cmdLoading.value = false
  }
}

async function handleEquipmentChange() {
  loadTelemetry()
  loadCommands()
}

async function refreshData() {
  loadTelemetry()
  loadCommands()
}

async function sendCommand() {
  if (!selectedEquipment.value) {
    ElMessage.warning(t('mes.scada.selectEquipmentHint'))
    return
  }
  cmdSending.value = true
  try {
    await dispatchCommandAPI({
      equipmentCode: selectedEquipment.value,
      commandType: commandForm.value.commandType,
      payload: commandForm.value.payload,
      createdBy: 'admin',
    })
    ElMessage.success(t('mes.scada.commandSent'))
    commandForm.value.payload = ''
    loadCommands()
  } catch {
    ElMessage.error(t('mes.scada.commandFailed'))
  } finally {
    cmdSending.value = false
  }
}

onMounted(() => {
  loadEquipment()
})
</script>

<style scoped>
.metric-card {
  text-align: center;
  border: 1px solid #ebeef5;
  transition: all 0.3s;
}
.metric-card:hover {
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}
.metric-alert {
  border-color: #f56c6c;
  background-color: #fef0f0;
}
.metric-name {
  font-size: 13px;
  color: #909399;
  margin-bottom: 4px;
}
.metric-value {
  font-size: 24px;
  font-weight: bold;
}
.metric-unit {
  font-size: 13px;
  font-weight: normal;
  color: #909399;
}
.metric-range {
  font-size: 11px;
  color: #c0c4cc;
  margin-top: 2px;
}
.command-item {
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}
.command-item:last-child {
  border-bottom: none;
}
.command-header {
  margin-bottom: 4px;
}
.command-payload {
  font-size: 12px;
  color: #606266;
  word-break: break-all;
}
.command-time {
  font-size: 11px;
  color: #c0c4cc;
  margin-top: 2px;
}
.command-list-header {
  margin-bottom: 8px;
}
</style>
