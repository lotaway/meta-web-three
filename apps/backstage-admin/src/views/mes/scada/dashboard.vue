<template>
  <div class="app-container">
    <div class="filter-container">
      <el-select v-model="workshopId" filterable :placeholder="t('mes.scada.selectWorkshop')"
        style="width: 220px" class="filter-item" @change="refreshAll">
        <el-option :value="" :label="t('mes.scada.all')" />
        <el-option v-for="ws in workshopList" :key="ws.id" :label="ws.workshopName" :value="ws.id" />
      </el-select>
      <el-button type="primary" :icon="Refresh" class="filter-item" @click="refreshAll">
        {{ t('common.refresh') }}
      </el-button>
      <el-tag type="info" effect="dark" class="filter-item">
        {{ t('mes.scada.lastUpdate') }}: {{ lastUpdateTime }}
      </el-tag>
    </div>

    <el-row :gutter="16">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #e6f7ff;">
            <el-icon color="#1890ff"><Monitor /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ metrics.totalEquipment }}</div>
            <div class="stat-label">{{ t('mes.scada.totalEquipment') }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #f6ffed;">
            <el-icon color="#52c41a"><CheckCircle /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ metrics.onlineEquipment }}</div>
            <div class="stat-label">{{ t('mes.scada.onlineEquipment') }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" :style="oeeColorStyle">
            <el-icon :color="oeeColor"><TrendingUp /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ metrics.avgOee }}%</div>
            <div class="stat-label">{{ t('mes.scada.avgOee') }}</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #fff7e6;">
            <el-icon color="#fa8c16"><ShoppingCart /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ metrics.todayOutput }}</div>
            <div class="stat-label">{{ t('mes.scada.todayOutput') }}</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px;">
      <el-col :span="16">
        <el-card shadow="hover">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><Monitor /></el-icon>
              {{ t('mes.scada.equipmentStatus') }}
            </span>
          </template>
          <el-table v-loading="equipmentLoading" :data="equipmentList" size="small" border style="width: 100%">
            <el-table-column :label="t('mes.scada.equipmentCode')" prop="equipmentCode" width="120" />
            <el-table-column :label="t('mes.scada.equipmentName')" prop="equipmentName" width="150" />
            <el-table-column :label="t('mes.scada.status')" prop="status" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusTagType(row.status)" size="small">
                  {{ t(`mes.equipment.status${row.status}`) || row.status }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.scada.oee')" prop="oee" width="100">
              <template #default="{ row }">
                <span :style="{ color: getOeeColor(row.oee) }">
                  {{ row.oee != null ? row.oee + '%' : '-' }}
                </span>
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.scada.todayOutput')" prop="todayOutput" width="100" />
            <el-table-column :label="t('mes.scada.lastHeartbeat')" prop="lastHeartbeat" width="150" />
            <el-table-column :label="t('mes.scada.currentTask')" prop="currentTaskNo" />
          </el-table>
        </el-card>

        <el-card shadow="hover" style="margin-top: 16px;">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><List /></el-icon>
              {{ t('mes.scada.productionProgress') }}
            </span>
          </template>
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item :label="t('mes.scada.totalWorkOrders')">
              {{ productionStats.totalWorkOrders }}
            </el-descriptions-item>
            <el-descriptions-item :label="t('mes.scada.inProgressOrders')">
              {{ productionStats.inProgressWorkOrders }}
            </el-descriptions-item>
            <el-descriptions-item :label="t('mes.scada.completedOrders')">
              {{ productionStats.completedWorkOrders }}
            </el-descriptions-item>
            <el-descriptions-item :label="t('mes.scada.pendingOrders')">
              {{ productionStats.pendingWorkOrders }}
            </el-descriptions-item>
            <el-descriptions-item :label="t('mes.scada.todayOutput')" :span="2">
              {{ productionStats.todayOutput }} / {{ productionStats.todayPlannedOutput }}
              <el-progress :percentage="productionStats.completionRate" :stroke-width="8" style="margin-top: 8px;" />
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><Bell /></el-icon>
              {{ t('mes.scada.activeAlerts') }}
              <el-tag v-if="metrics.activeAlerts > 0" type="danger" style="margin-left: 8px;">
                {{ metrics.activeAlerts }}
              </el-tag>
            </span>
          </template>
          <div v-loading="alertsLoading">
            <div v-for="alert in alerts" :key="alert.id" class="alert-item">
              <div class="alert-header">
                <el-tag :type="getAlertLevelType(alert.levelCode)" size="small">
                  {{ alert.levelName }}
                </el-tag>
                <span class="alert-time">{{ alert.occurredAt }}</span>
              </div>
              <div class="alert-equipment">{{ alert.equipmentName || alert.equipmentCode }}</div>
              <div class="alert-desc">{{ alert.description }}</div>
              <div class="alert-footer">
                <span class="alert-no">{{ alert.eventNo }}</span>
                <span v-if="alert.reporterName" class="alert-reporter">{{ alert.reporterName }}</span>
              </div>
            </div>
            <el-empty v-if="alerts.length === 0" :description="t('mes.scada.noAlerts')" />
          </div>
        </el-card>

        <el-card shadow="hover" style="margin-top: 16px;">
          <template #header>
            <span>
              <el-icon style="vertical-align: middle; margin-right: 4px;"><PieChart /></el-icon>
              {{ t('mes.scada.equipmentStatusDistribution') }}
            </span>
          </template>
          <div class="status-dist">
            <div class="status-item">
              <div class="status-dot running"></div>
              <span>{{ t('mes.scada.running') }}</span>
              <span class="status-count">{{ metrics.runningEquipment }}</span>
            </div>
            <div class="status-item">
              <div class="status-dot idle"></div>
              <span>{{ t('mes.scada.idle') }}</span>
              <span class="status-count">{{ metrics.idleEquipment }}</span>
            </div>
            <div class="status-item">
              <div class="status-dot warning"></div>
              <span>{{ t('mes.scada.warning') }}</span>
              <span class="status-count">{{ metrics.warningEquipment }}</span>
            </div>
            <div class="status-item">
              <div class="status-dot error"></div>
              <span>{{ t('mes.scada.error') }}</span>
              <span class="status-count">{{ metrics.errorEquipment }}</span>
            </div>
            <div class="status-item">
              <div class="status-dot offline"></div>
              <span>{{ t('mes.scada.offline') }}</span>
              <span class="status-count">{{ metrics.offlineEquipment }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { Refresh, Monitor, CheckCircle, TrendingUp, ShoppingCart, Bell, List, PieChart } from '@element-plus/icons-vue'
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  getDashboardMetricsAPI,
  getEquipmentStatusSummaryAPI,
  getActiveAlertsAPI,
  getProductionStatsAPI,
  getWorkshopListAPI,
  type DashboardMetrics,
  type EquipmentStatusSummary,
  type AlertSummary,
  type ProductionStats,
  type Workshop,
} from '@/apis/scada'

const { t } = useI18n()

const workshopId = ref('')
const workshopList = ref<Workshop[]>([])

const metrics = ref<DashboardMetrics>({
  totalEquipment: 0, onlineEquipment: 0, runningEquipment: 0, idleEquipment: 0,
  warningEquipment: 0, errorEquipment: 0, offlineEquipment: 0,
  avgOee: 0, todayOutput: 0, activeAlerts: 0, pendingWorkOrders: 0,
})

const equipmentList = ref<EquipmentStatusSummary[]>([])
const alerts = ref<AlertSummary[]>([])
const productionStats = ref<ProductionStats>({
  totalWorkOrders: 0, inProgressWorkOrders: 0, completedWorkOrders: 0, pendingWorkOrders: 0,
  todayOutput: 0, todayPlannedOutput: 0, totalTasks: 0, completedTasks: 0, completionRate: 0,
})

const metricsLoading = ref(false)
const equipmentLoading = ref(false)
const alertsLoading = ref(false)
const productionLoading = ref(false)
const lastUpdateTime = ref('-')

let refreshTimer: number | null = null

const oeeColor = computed(() => {
  const oee = metrics.value.avgOee
  if (oee >= 85) return '#52c41a'
  if (oee >= 60) return '#faad14'
  return '#f5222d'
})

const oeeColorStyle = computed(() => ({
  backgroundColor: oeeColor.value === '#52c41a' ? '#f6ffed' : oeeColor.value === '#faad14' ? '#fff7e6' : '#fff2f0',
}))

function getStatusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info'> = {
    RUNNING: 'success', IDLE: 'info', ONLINE: 'success', WARNING: 'warning', ERROR: 'danger', OFFLINE: 'danger',
  }
  return map[status] || 'info'
}

function getOeeColor(oee: number): string {
  if (oee == null) return '#909399'
  if (oee >= 85) return '#52c41a'
  if (oee >= 60) return '#faad14'
  return '#f5222d'
}

function getAlertLevelType(levelCode: string): 'success' | 'warning' | 'danger' | 'info' {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info'> = {
    INFO: 'info', WARNING: 'warning', ERROR: 'danger', CRITICAL: 'danger',
  }
  return map[levelCode] || 'info'
}

async function loadMetrics() {
  metricsLoading.value = true
  try {
    const res = await getDashboardMetricsAPI(workshopId.value || undefined)
    metrics.value = res.data || metrics.value
  } catch {
    // silent
  } finally {
    metricsLoading.value = false
  }
}

async function loadEquipment() {
  equipmentLoading.value = true
  try {
    const res = await getEquipmentStatusSummaryAPI(workshopId.value || undefined)
    equipmentList.value = res.data || []
  } catch {
    // silent
  } finally {
    equipmentLoading.value = false
  }
}

async function loadAlerts() {
  alertsLoading.value = true
  try {
    const res = await getActiveAlertsAPI(workshopId.value || undefined)
    alerts.value = (res.data || []).slice(0, 10)
  } catch {
    // silent
  } finally {
    alertsLoading.value = false
  }
}

async function loadProductionStats() {
  productionLoading.value = true
  try {
    const res = await getProductionStatsAPI(workshopId.value || undefined)
    productionStats.value = res.data || productionStats.value
  } catch {
    // silent
  } finally {
    productionLoading.value = false
  }
}

async function loadWorkshops() {
  try {
    const res = await getWorkshopListAPI()
    workshopList.value = res.data || []
  } catch {
    // silent
  }
}

async function refreshAll() {
  await Promise.all([loadMetrics(), loadEquipment(), loadAlerts(), loadProductionStats()])
  lastUpdateTime.value = new Date().toLocaleString()
}

onMounted(() => {
  loadWorkshops()
  refreshAll()
  refreshTimer = window.setInterval(refreshAll, 30000)
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})
</script>

<style scoped>
.stat-card {
  display: flex;
  align-items: center;
  padding: 16px;
}
.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
}
.stat-icon .el-icon {
  font-size: 24px;
}
.stat-info {
  flex: 1;
}
.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}
.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
}

.alert-item {
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
  background: #fffbe6;
  margin-bottom: 8px;
  border-radius: 4px;
}
.alert-item:last-child {
  margin-bottom: 0;
}
.alert-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.alert-time {
  font-size: 11px;
  color: #909399;
}
.alert-equipment {
  font-size: 13px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 4px;
}
.alert-desc {
  font-size: 12px;
  color: #606266;
  margin-bottom: 4px;
}
.alert-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.alert-no {
  font-size: 11px;
  color: #c0c4cc;
}
.alert-reporter {
  font-size: 11px;
  color: #909399;
}

.status-dist {
  padding: 8px 0;
}
.status-item {
  display: flex;
  align-items: center;
  padding: 6px 0;
}
.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
}
.status-dot.running { background-color: #52c41a; }
.status-dot.idle { background-color: #1890ff; }
.status-dot.warning { background-color: #faad14; }
.status-dot.error { background-color: #f5222d; }
.status-dot.offline { background-color: #909399; }
.status-count {
  margin-left: auto;
  font-weight: bold;
  color: #303133;
}
</style>
