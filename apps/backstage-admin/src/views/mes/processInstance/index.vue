<template>
  <div class="process-instance-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="filterForm" class="filter-form">
        <el-form-item :label="t('mes.processInstance.businessType')">
          <el-select
            v-model="filterForm.businessType"
            :placeholder="t('mes.processInstance.businessTypePlaceholder')"
            clearable
            @change="handleFilterChange"
          >
            <el-option label="" value="" />
            <el-option :label="t('mes.processInstance.typeWorkOrder')" value="WORK_ORDER" />
            <el-option :label="t('mes.processInstance.typeProductionTask')" value="PRODUCTION_TASK" />
            <el-option :label="t('mes.processInstance.typeQcInspection')" value="QC_INSPECTION" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.processInstance.status')">
          <el-select
            v-model="filterForm.status"
            :placeholder="t('mes.processInstance.statusPlaceholder')"
            clearable
            @change="handleFilterChange"
          >
            <el-option label="" value="" />
            <el-option :label="t('mes.processInstance.statusRunning')" value="RUNNING" />
            <el-option :label="t('mes.processInstance.statusCompleted')" value="COMPLETED" />
            <el-option :label="t('mes.processInstance.statusTerminated')" value="TERMINATED" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleFilterChange">
            {{ t('mes.processInstance.search') }}
          </el-button>
          <el-button @click="handleReset">{{ t('mes.processInstance.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.processInstance.title') }}</span>
        </div>
      </template>

      <el-table :data="tableData" v-loading="loading" border stripe>
        <el-table-column prop="id" :label="t('mes.processInstance.id')" width="80" />
        <el-table-column prop="templateName" :label="t('mes.processInstance.templateName')" min-width="150" />
        <el-table-column prop="businessType" :label="t('mes.processInstance.businessType')" width="150">
          <template #default="{ row }">
            {{ getBusinessTypeText(row.businessType) }}
          </template>
        </el-table-column>
        <el-table-column prop="businessKey" :label="t('mes.processInstance.businessKey')" width="150" />
        <el-table-column prop="status" :label="t('mes.processInstance.status')" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="currentNodeId" :label="t('mes.processInstance.currentNode')" width="150" />
        <el-table-column prop="startedAt" :label="t('mes.processInstance.startedAt')" width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.startedAt) }}
          </template>
        </el-table-column>
        <el-table-column prop="completedAt" :label="t('mes.processInstance.completedAt')" width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.completedAt) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processInstance.actions')" width="180" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">
              {{ t('mes.processInstance.view') }}
            </el-button>
            <el-button link type="success" size="small" @click="handleComplete(row)" v-if="row.status === 'RUNNING'">
              {{ t('mes.processInstance.complete') }}
            </el-button>
            <el-button link type="danger" size="small" @click="handleTerminate(row)" v-if="row.status === 'RUNNING'">
              {{ t('mes.processInstance.terminate') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-container">
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :total="pagination.total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listProcessInstancesAPI,
  completeProcessInstanceAPI,
  terminateProcessInstanceAPI,
} from '@/apis/processFlow'
import type { ProcessFlowInstance } from '@/apis/processFlow'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const tableData = ref<ProcessFlowInstance[]>([])

const filterForm = reactive({
  businessType: '',
  status: '',
})

const pagination = reactive({
  page: 1,
  pageSize: 10,
  total: 0,
})

const getStatusType = (status?: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const statusMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    RUNNING: 'success',
    COMPLETED: 'info',
    TERMINATED: 'danger',
  }
  return statusMap[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const textMap: Record<string, string> = {
    RUNNING: t('mes.processInstance.statusRunning'),
    COMPLETED: t('mes.processInstance.statusCompleted'),
    TERMINATED: t('mes.processInstance.statusTerminated'),
  }
  return textMap[status || ''] || status
}

const getBusinessTypeText = (type?: string) => {
  const textMap: Record<string, string> = {
    WORK_ORDER: t('mes.processInstance.typeWorkOrder'),
    PRODUCTION_TASK: t('mes.processInstance.typeProductionTask'),
    QC_INSPECTION: t('mes.processInstance.typeQcInspection'),
  }
  return textMap[type || ''] || type
}

const formatDateTime = (dateStr?: string) => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

const fetchData = async () => {
  loading.value = true
  try {
    const res = await listProcessInstancesAPI(
      filterForm.businessType || undefined,
      filterForm.status || undefined
    )
    tableData.value = res.data || []
    pagination.total = tableData.value.length
  } catch (error) {
    ElMessage.error(t('mes.processInstance.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleFilterChange = () => {
  pagination.page = 1
  fetchData()
}

const handleReset = () => {
  filterForm.businessType = ''
  filterForm.status = ''
  handleFilterChange()
}

const handleView = (row: ProcessFlowInstance) => {
  router.push(`/mes/process-instance/detail?id=${row.id}`)
}

const handleComplete = async (row: ProcessFlowInstance) => {
  try {
    await ElMessageBox.confirm(
      t('mes.processInstance.confirmComplete'),
      t('mes.processInstance.warning'),
      { confirmButtonText: t('mes.processInstance.confirm'), cancelButtonText: t('mes.processInstance.cancel'), type: 'warning' }
    )
    await completeProcessInstanceAPI(row.id!, 1)
    ElMessage.success(t('mes.processInstance.completeSuccess'))
    fetchData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processInstance.completeFailed'))
    }
  }
}

const handleTerminate = async (row: ProcessFlowInstance) => {
  try {
    await ElMessageBox.confirm(
      t('mes.processInstance.confirmTerminate'),
      t('mes.processInstance.warning'),
      { confirmButtonText: t('mes.processInstance.confirm'), cancelButtonText: t('mes.processInstance.cancel'), type: 'warning' }
    )
    await terminateProcessInstanceAPI(row.id!, 1)
    ElMessage.success(t('mes.processInstance.terminateSuccess'))
    fetchData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processInstance.terminateFailed'))
    }
  }
}

const handleSizeChange = (size: number) => {
  pagination.pageSize = size
  fetchData()
}

const handlePageChange = (page: number) => {
  pagination.page = page
  fetchData()
}

onMounted(() => {
  fetchData()
})
</script>

<style scoped>
.process-instance-container {
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

.table-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
</style>