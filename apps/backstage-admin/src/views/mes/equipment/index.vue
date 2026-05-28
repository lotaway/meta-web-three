<template>
  <div class="equipment-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.equipment.equipmentCode')">
          <el-input v-model="queryParams.equipmentCode" :placeholder="t('mes.equipment.equipmentCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.equipmentName')">
          <el-input v-model="queryParams.equipmentName" :placeholder="t('mes.equipment.equipmentNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.equipment.statusPlaceholder')" clearable>
            <el-option :label="t('mes.equipment.statusIdle')" value="IDLE" />
            <el-option :label="t('mes.equipment.statusRunning')" value="RUNNING" />
            <el-option :label="t('mes.equipment.statusBreakdown')" value="BREAKDOWN" />
            <el-option :label="t('mes.equipment.statusMaintenance')" value="MAINTENANCE" />
            <el-option :label="t('mes.equipment.statusOffline')" value="OFFLINE" />
            <el-option :label="t('mes.equipment.statusOnline')" value="ONLINE" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
          <el-button @click="resetQuery">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">{{ t('mes.equipment.add') }}</el-button>
      </div>

      <el-table :data="equipmentList" v-loading="loading" border stripe>
        <el-table-column :label="t('common.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.equipment.equipmentCode')" prop="equipmentCode" width="150" />
        <el-table-column :label="t('mes.equipment.equipmentName')" prop="equipmentName" width="180" />
        <el-table-column :label="t('mes.equipment.equipmentTypeCode')" prop="equipmentTypeCode" width="120" />
        <el-table-column :label="t('mes.equipment.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.equipment.todayOutput')" prop="todayOutput" width="100" />
        <el-table-column :label="t('mes.equipment.oee')" width="100">
          <template #default="{ row }">
            {{ row.utilizationRate?.toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.equipment.ipAddress')" prop="ipAddress" width="140" />
        <el-table-column :label="t('mes.equipment.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operation')" fixed="right" width="280">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleStartTask(row)" v-if="row.status === 'IDLE'">{{ t('mes.equipment.startTask') }}</el-button>
            <el-button link type="warning" size="small" @click="handleReportBreakdown(row)" v-if="row.status === 'RUNNING'">{{ t('mes.equipment.reportBreakdown') }}</el-button>
            <el-button link type="info" size="small" @click="handleMaintenance(row)" v-if="row.status === 'IDLE'">{{ t('mes.equipment.startMaintenance') }}</el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="queryParams.pageNum"
        v-model:page-size="queryParams.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { Equipment } from '@/apis/equipment'
import { 
  getEquipmentListAPI, 
  deleteEquipmentAPI,
  startTaskAPI,
  reportBreakdownAPI,
  startMaintenanceAPI 
} from '@/apis/equipment'

const { t } = useI18n()

const router = useRouter()

const loading = ref(false)
const equipmentList = ref<Equipment[]>([])
const total = ref(0)

const queryParams = reactive({
  equipmentCode: '',
  equipmentName: '',
  status: '',
  pageNum: 1,
  pageSize: 10
})

const getList = async () => {
  loading.value = true
  try {
    const data = await getEquipmentListAPI({
      status: queryParams.status as any || undefined
    })
    equipmentList.value = data || []
    total.value = data?.length || 0
  } catch (error) {
    console.error(t('mes.equipment.fetchListFailed'), error)
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.equipmentCode = ''
  queryParams.equipmentName = ''
  queryParams.status = ''
  handleQuery()
}

const handleAdd = () => {
  router.push('/mes/equipment/form')
}

const handleEdit = (row: Equipment) => {
  router.push({ path: '/mes/equipment/form', query: { id: row.id } })
}

const handleView = (row: Equipment) => {
  router.push({ path: '/mes/equipment/detail', query: { id: row.id } })
}

const handleStartTask = async (row: Equipment) => {
  try {
    const taskNo = await ElMessageBox.prompt(t('mes.equipment.taskNoPlaceholder'), t('mes.equipment.startTask'), {
      confirmButtonText: t('common.confirmText'),
      cancelButtonText: t('common.cancelText'),
      inputPattern: /.+/,
      inputErrorMessage: t('mes.equipment.taskNoError')
    })
    if (taskNo.value) {
      await startTaskAPI(row.id!, taskNo.value)
      ElMessage.success(t('mes.equipment.taskStartSuccess'))
      getList()
    }
  } catch (error) {
    // 用户取消
  }
}

const handleReportBreakdown = async (row: Equipment) => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.confirmBreakdown'), t('common.warning'), {
      type: 'warning'
    })
    await reportBreakdownAPI(row.id!)
    ElMessage.success(t('mes.equipment.breakdownReported'))
    getList()
  } catch (error) {
    // 用户取消
  }
}

const handleMaintenance = async (row: Equipment) => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.confirmMaintenance'), t('common.warning'), {
      type: 'warning'
    })
    await startMaintenanceAPI(row.id!)
    ElMessage.success(t('mes.equipment.maintenanceStarted'))
    getList()
  } catch (error) {
    // 用户取消
  }
}

const handleDelete = async (row: Equipment) => {
  try {
    await ElMessageBox.confirm(t('mes.equipment.confirmDelete'), t('common.warning'), {
      type: 'warning'
    })
    await deleteEquipmentAPI(row.id!)
    ElMessage.success(t('mes.equipment.deleteSuccess'))
    getList()
  } catch (error) {
    // 用户取消
  }
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

onMounted(() => {
  getList()
})
</script>

<style scoped>
.equipment-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.toolbar {
  margin-bottom: 15px;
}
</style>