<template>
  <div class="equipment-detail-container">
    <el-card v-loading="loading">
      <template #header>
        <div class="header-row">
          <span>设备详情</span>
          <el-button @click="handleBack">返回</el-button>
        </div>
      </template>
      
      <el-descriptions :column="2" border v-if="equipment">
        <el-descriptions-item label="设备编码">{{ equipment.equipmentCode }}</el-descriptions-item>
        <el-descriptions-item label="设备名称">{{ equipment.equipmentName }}</el-descriptions-item>
        <el-descriptions-item label="设备类型">{{ equipment.equipmentTypeCode }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="getStatusType(equipment.status)">{{ getStatusText(equipment.status) }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="车间ID">{{ equipment.workshopId }}</el-descriptions-item>
        <el-descriptions-item label="工位ID">{{ equipment.workstationId }}</el-descriptions-item>
        <el-descriptions-item label="当前位置">
          X: {{ equipment.positionX }}, Y: {{ equipment.positionY }}, Z: {{ equipment.positionZ }}
        </el-descriptions-item>
        <el-descriptions-item label="IP地址">{{ equipment.ipAddress }}</el-descriptions-item>
        <el-descriptions-item label="MAC地址">{{ equipment.macAddress }}</el-descriptions-item>
        <el-descriptions-item label="数字孪生设备">{{ equipment.digitalTwinDeviceCode || '-' }}</el-descriptions-item>
        <el-descriptions-item label="今日产出">{{ equipment.todayOutput }}</el-descriptions-item>
        <el-descriptions-item label="OEE">
          {{ equipment.utilizationRate?.toFixed(2) }}%
        </el-descriptions-item>
        <el-descriptions-item label="运行时间">{{ formatDuration(equipment.totalRunningSeconds) }}</el-descriptions-item>
        <el-descriptions-item label="空闲时间">{{ formatDuration(equipment.totalIdleSeconds) }}</el-descriptions-item>
        <el-descriptions-item label="停机时间">{{ formatDuration(equipment.totalDowntimeSeconds) }}</el-descriptions-item>
        <el-descriptions-item label="当前任务">{{ equipment.currentTaskNo || '-' }}</el-descriptions-item>
        <el-descriptions-item label="最后保养时间">{{ equipment.lastMaintenanceTime || '-' }}</el-descriptions-item>
        <el-descriptions-item label="下次保养时间">{{ equipment.nextMaintenanceTime || '-' }}</el-descriptions-item>
        <el-descriptions-item label="最后心跳">{{ equipment.lastHeartbeat || '-' }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ equipment.createdAt }}</el-descriptions-item>
      </el-descriptions>
      
      <el-divider />
      
      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit" v-if="equipment">编辑</el-button>
        <el-button type="success" @click="handleStartTask" v-if="equipment?.status === 'IDLE'">开始任务</el-button>
        <el-button type="info" @click="handleCompleteTask" v-if="equipment?.status === 'RUNNING'">完成任务</el-button>
        <el-button type="warning" @click="handleReportBreakdown" v-if="equipment?.status === 'RUNNING'">报告故障</el-button>
        <el-button type="danger" @click="handleRepair" v-if="equipment?.status === 'BREAKDOWN'">维修</el-button>
        <el-button @click="handleStartMaintenance" v-if="equipment?.status === 'IDLE'">开始保养</el-button>
        <el-button @click="handleCompleteMaintenance" v-if="equipment?.status === 'MAINTENANCE'">完成保养</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
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

const router = useRouter()
const route = useRoute()
const loading = ref(false)
const equipment = ref<Equipment | null>(null)

const loadData = async () => {
  if (!route.query.id) return
  
  const id = Number(route.query.id)
  loading.value = true
  try {
    equipment.value = await getEquipmentByIdAPI(id)
  } catch (error) {
    ElMessage.error('加载设备信息失败')
  } finally {
    loading.value = false
  }
}

const handleEdit = () => {
  router.push({ path: '/mes/equipment/form', query: { id: route.query.id } })
}

const handleStartTask = async () => {
  try {
    const taskNo = await ElMessageBox.prompt('请输入任务编号', '开始任务', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      inputPattern: /.+/,
      inputErrorMessage: '请输入任务编号'
    })
    if (taskNo.value && equipment.value) {
      equipment.value = await startTaskAPI(equipment.value.id!, taskNo.value)
      ElMessage.success('任务已开始')
    }
  } catch (error) {
    // 用户取消
  }
}

const handleCompleteTask = async () => {
  try {
    await ElMessageBox.confirm('确定要完成当前任务吗？', '提示', { type: 'warning' })
    if (equipment.value) {
      equipment.value = await completeTaskAPI(equipment.value.id!)
      ElMessage.success('任务已完成')
    }
  } catch (error) {
    // 用户取消
  }
}

const handleReportBreakdown = async () => {
  try {
    await ElMessageBox.confirm('确定要报告设备故障吗？', '提示', { type: 'warning' })
    if (equipment.value) {
      equipment.value = await reportBreakdownAPI(equipment.value.id!)
      ElMessage.success('已报告故障')
    }
  } catch (error) {
    // 用户取消
  }
}

const handleRepair = async () => {
  try {
    await ElMessageBox.confirm('确定要维修该设备吗？', '提示', { type: 'warning' })
    if (equipment.value) {
      equipment.value = await repairEquipmentAPI(equipment.value.id!)
      ElMessage.success('维修完成')
    }
  } catch (error) {
    // 用户取消
  }
}

const handleStartMaintenance = async () => {
  try {
    await ElMessageBox.confirm('确定要开始设备保养吗？', '提示', { type: 'warning' })
    if (equipment.value) {
      equipment.value = await startMaintenanceAPI(equipment.value.id!)
      ElMessage.success('已启动保养')
    }
  } catch (error) {
    // 用户取消
  }
}

const handleCompleteMaintenance = async () => {
  try {
    await ElMessageBox.confirm('确定要完成保养吗？', '提示', { type: 'warning' })
    if (equipment.value) {
      equipment.value = await completeMaintenanceAPI(equipment.value.id!)
      ElMessage.success('保养已完成')
    }
  } catch (error) {
    // 用户取消
  }
}

const handleBack = () => {
  router.back()
}

const getStatusType = (status?: string) => {
  const map: Record<string, string> = {
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
    IDLE: '空闲',
    RUNNING: '运行中',
    BREAKDOWN: '故障',
    MAINTENANCE: '保养中',
    OFFLINE: '离线',
    ONLINE: '在线',
    WARNING: '警告',
    ERROR: '错误'
  }
  return map[status || ''] || status
}

const formatDuration = (seconds?: number) => {
  if (!seconds) return '0秒'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  return `${hours}小时${minutes}分钟${secs}秒`
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