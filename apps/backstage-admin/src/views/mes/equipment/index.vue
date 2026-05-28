<template>
  <div class="equipment-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item label="设备编码">
          <el-input v-model="queryParams.equipmentCode" placeholder="请输入设备编码" clearable />
        </el-form-item>
        <el-form-item label="设备名称">
          <el-input v-model="queryParams.equipmentName" placeholder="请输入设备名称" clearable />
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="queryParams.status" placeholder="请选择状态" clearable>
            <el-option label="空闲" value="IDLE" />
            <el-option label="运行中" value="RUNNING" />
            <el-option label="故障" value="BREAKDOWN" />
            <el-option label="保养中" value="MAINTENANCE" />
            <el-option label="离线" value="OFFLINE" />
            <el-option label="在线" value="ONLINE" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">查询</el-button>
          <el-button @click="resetQuery">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">新增设备</el-button>
      </div>

      <el-table :data="equipmentList" v-loading="loading" border stripe>
        <el-table-column label="ID" prop="id" width="80" />
        <el-table-column label="设备编码" prop="equipmentCode" width="150" />
        <el-table-column label="设备名称" prop="equipmentName" width="180" />
        <el-table-column label="设备类型" prop="equipmentTypeCode" width="120" />
        <el-table-column label="状态" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="今日产出" prop="todayOutput" width="100" />
        <el-table-column label="OEE" width="100">
          <template #default="{ row }">
            {{ row.utilizationRate?.toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column label="IP地址" prop="ipAddress" width="140" />
        <el-table-column label="创建时间" prop="createdAt" width="180" />
        <el-table-column label="操作" fixed="right" width="280">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">查看</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">编辑</el-button>
            <el-button link type="success" size="small" @click="handleStartTask(row)" v-if="row.status === 'IDLE'">开始任务</el-button>
            <el-button link type="warning" size="small" @click="handleReportBreakdown(row)" v-if="row.status === 'RUNNING'">报故障</el-button>
            <el-button link type="info" size="small" @click="handleMaintenance(row)" v-if="row.status === 'IDLE'">保养</el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">删除</el-button>
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
import { ElMessage, ElMessageBox } from 'element-plus'
import type { Equipment } from '@/apis/equipment'
import { 
  getEquipmentListAPI, 
  deleteEquipmentAPI,
  startTaskAPI,
  reportBreakdownAPI,
  startMaintenanceAPI 
} from '@/apis/equipment'

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
    console.error('获取设备列表失败:', error)
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
    const taskNo = await ElMessageBox.prompt('请输入任务编号', '开始任务', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      inputPattern: /.+/,
      inputErrorMessage: '请输入任务编号'
    })
    if (taskNo.value) {
      await startTaskAPI(row.id!, taskNo.value)
      ElMessage.success('任务已开始')
      getList()
    }
  } catch (error) {
    // 用户取消
  }
}

const handleReportBreakdown = async (row: Equipment) => {
  try {
    await ElMessageBox.confirm('确定要报告该设备故障吗？', '提示', {
      type: 'warning'
    })
    await reportBreakdownAPI(row.id!)
    ElMessage.success('已报告故障')
    getList()
  } catch (error) {
    // 用户取消
  }
}

const handleMaintenance = async (row: Equipment) => {
  try {
    await ElMessageBox.confirm('确定要开始设备保养吗？', '提示', {
      type: 'warning'
    })
    await startMaintenanceAPI(row.id!)
    ElMessage.success('已启动保养')
    getList()
  } catch (error) {
    // 用户取消
  }
}

const handleDelete = async (row: Equipment) => {
  try {
    await ElMessageBox.confirm('确定要删除该设备吗？', '提示', {
      type: 'warning'
    })
    await deleteEquipmentAPI(row.id!)
    ElMessage.success('删除成功')
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