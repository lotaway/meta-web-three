<template>
  <div class="equipment-detail-container">
    <el-card v-loading="loading">
      <template #header>
        <div class="header-row">
          <span>{{ t('mes.equipment.detail.title') }}</span>
          <el-button @click="handleBack">{{ t('mes.equipment.detail.back') }}</el-button>
        </div>
      </template>
      
      <el-descriptions :column="2" border v-if="equipment">
        <el-descriptions-item :label="t('mes.equipment.equipmentCode')">{{ equipment.equipmentCode }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.equipmentName')">{{ equipment.equipmentName }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.equipmentType')">{{ equipment.equipmentTypeCode }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.status')">
          <el-tag :type="getStatusType(equipment.status)">{{ getStatusText(equipment.status) }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.workshopId')">{{ equipment.workshopId }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.workstationId')">{{ equipment.workstationId }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.detail.currentPosition')">
          X: {{ equipment.positionX }}, Y: {{ equipment.positionY }}, Z: {{ equipment.positionZ }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.ipAddress')">{{ equipment.ipAddress }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.macAddress')">{{ equipment.macAddress }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.digitalTwin')">{{ equipment.digitalTwinDeviceCode || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.todayOutput')">{{ equipment.todayOutput }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.oee')">
          {{ equipment.utilizationRate?.toFixed(2) }}%
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.runningTime')">{{ formatDuration(equipment.totalRunningSeconds) }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.idleTime')">{{ formatDuration(equipment.totalIdleSeconds) }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.downtime')">{{ formatDuration(equipment.totalDowntimeSeconds) }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.currentTask')">{{ equipment.currentTaskNo || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.lastMaintenance')">{{ equipment.lastMaintenanceTime || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.nextMaintenance')">{{ equipment.nextMaintenanceTime || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.lastHeartbeat')">{{ equipment.lastHeartbeat || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.createdAt')">{{ equipment.createdAt }}</el-descriptions-item>
      </el-descriptions>
      
      <el-divider />
      
      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit" v-if="equipment">{{ t('mes.equipment.detail.edit') }}</el-button>
        <el-button type="success" @click="handleStartTask" v-if="equipment?.status === 'IDLE'">{{ t('mes.equipment.detail.startTask') }}</el-button>
        <el-button type="info" @click="handleCompleteTask" v-if="equipment?.status === 'RUNNING'">{{ t('mes.equipment.detail.completeTask') }}</el-button>
        <el-button type="warning" @click="handleReportBreakdown" v-if="equipment?.status === 'RUNNING'">{{ t('mes.equipment.detail.reportBreakdown') }}</el-button>
        <el-button type="danger" @click="handleRepair" v-if="equipment?.status === 'BREAKDOWN'">{{ t('mes.equipment.detail.repair') }}</el-button>
        <el-button @click="handleStartMaintenance" v-if="equipment?.status === 'IDLE'">{{ t('mes.equipment.detail.startMaintenance') }}</el-button>
        <el-button @click="handleCompleteMaintenance" v-if="equipment?.status === 'MAINTENANCE'">{{ t('mes.equipment.detail.completeMaintenance') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { Equipment } from '@/apis/equipment'
import { 
  getEquipmentByIdAPI,
  startTaskAPI,
  completeTaskAPI,
  reportBreakdownAPI,
  repairEquipmentAPI,
  startMaintenanceAPI,
  completeMaintenanceAPI
} from '@/apis/equipment'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const loading = ref(false)
const equipment = ref<Equipment | null>(null)

const loadData = async () => {
  if (!route.query.id) return
  
  const id = Number(route.query.id)
  loading.value = true
  try {
    const res = await getEquipmentByIdAPI(id)
    equipment.value = res.data
  } catch (error) {
    ElMessage.error(t('mes.equipment.detail.loadError'))
  } finally {
    loading.value = false
  }
}

const handleEdit = () => {
  router.push({ path: '/mes/equipment/form', query: { id: route.query.id } })
}

const handleStartTask = async () => {
  try {
    const taskNo = await ElMessageBox.prompt(t('mes.equipment.detail.enterTaskNo'), t('mes.equipment.detail.startTaskTitle'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      inputPattern: /.+/,
      inputErrorMessage: t('mes.equipment.detail.enterTaskNo')
    })
    if (taskNo.value && equipment.value) {
      const res = await startTaskAPI(equipment.value.id!, taskNo.value)
      equipment.value = res.data
      ElMessage.success(t('mes.equipment.detail.taskStarted'))
    }
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.equipment.detail.taskStartFailed'))
    }
  }
}

const handleCompleteTask = async () => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.detail.confirmCompleteTask'), t('common.warning'), { type: 'warning' })
    if (equipment.value) {
      const res = await completeTaskAPI(equipment.value.id!)
      equipment.value = res.data
      ElMessage.success(t('mes.equipment.detail.taskCompleted'))
    }
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.equipment.detail.taskCompleteFailed'))
    }
  }
}

const handleReportBreakdown = async () => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.detail.confirmBreakdown'), t('common.warning'), { type: 'warning' })
    if (equipment.value) {
      const res = await reportBreakdownAPI(equipment.value.id!)
      equipment.value = res.data
      ElMessage.success(t('mes.equipment.detail.breakdownReported'))
    }
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.equipment.detail.breakdownReportFailed'))
    }
  }
}

const handleRepair = async () => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.detail.confirmRepair'), t('common.warning'), { type: 'warning' })
    if (equipment.value) {
      const res = await repairEquipmentAPI(equipment.value.id!)
      equipment.value = res.data
      ElMessage.success(t('mes.equipment.detail.repairCompleted'))
    }
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.equipment.detail.repairFailed'))
    }
  }
}

const handleStartMaintenance = async () => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.detail.confirmMaintenance'), t('common.warning'), { type: 'warning' })
    if (equipment.value) {
      const res = await startMaintenanceAPI(equipment.value.id!)
      equipment.value = res.data
      ElMessage.success(t('mes.equipment.detail.maintenanceStarted'))
    }
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.equipment.detail.maintenanceStartFailed'))
    }
  }
}

const handleCompleteMaintenance = async () => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.detail.confirmCompleteMaintenance'), t('common.warning'), { type: 'warning' })
    if (equipment.value) {
      const res = await completeMaintenanceAPI(equipment.value.id!)
      equipment.value = res.data
      ElMessage.success(t('mes.equipment.detail.maintenanceCompleted'))
    }
  } catch (error) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.equipment.detail.maintenanceCompleteFailed'))
    }
  }
}

const handleBack = () => {
  router.back()
}

const getStatusType = (status?: string): 'success' | 'warning' | 'danger' | 'info' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | undefined> = {
    IDLE: 'info',
    RUNNING: 'success',
    BREAKDOWN: 'danger',
    MAINTENANCE: 'warning',
    OFFLINE: 'info',
    ONLINE: 'success',
    WARNING: 'warning',
    ERROR: 'danger'
  }
  return map[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const map: Record<string, string> = {
    IDLE: t('mes.equipment.statusIdle'),
    RUNNING: t('mes.equipment.statusRunning'),
    BREAKDOWN: t('mes.equipment.statusBreakdown'),
    MAINTENANCE: t('mes.equipment.statusMaintenance'),
    OFFLINE: t('mes.equipment.statusOffline'),
    ONLINE: t('mes.equipment.statusOnline'),
    WARNING: t('mes.equipment.statusWarning'),
    ERROR: t('mes.equipment.statusError')
  }
  return map[status || ''] || status
}

const formatDuration = (seconds?: number) => {
  if (!seconds) return `0${t('mes.equipment.timeFormat.minutes')}`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  return `${hours}${t('mes.equipment.timeFormat.hours')}${minutes}${t('mes.equipment.timeFormat.minutes')}${secs}${t('mes.equipment.timeFormat.seconds')}`
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.equipment-detail-container {
  padding: 20px;
}

.header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.action-buttons {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
</style>