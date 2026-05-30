<template>
  <div class="app-container">
    <div class="filter-container">
      <el-input
        v-model="listQuery.taskNo"
        :placeholder="t('mes.productionTask.taskNoPlaceholder')"
        class="filter-item"
        style="width: 200px"
        @keyup.enter="handleFilter"
      />
      <el-select
        v-model="listQuery.status"
        :placeholder="t('mes.productionTask.statusPlaceholder')"
        clearable
        class="filter-item"
        style="width: 150px"
      >
        <el-option
          v-for="item in statusOptions"
          :key="item.value"
          :label="t(item.label)"
          :value="item.value"
        />
      </el-select>
      <el-button class="filter-item" type="primary" :icon="Search" @click="handleFilter">
        {{ t('common.query') }}
      </el-button>
      <el-button class="filter-item" type="primary" :icon="Plus" @click="handleCreate">
        {{ t('mes.productionTask.add') }}
      </el-button>
    </div>

    <el-table v-loading="listLoading" :data="list" border style="width: 100%">
      <el-table-column :label="t('common.id')" prop="id" width="80" />
      <el-table-column :label="t('mes.productionTask.taskNo')" prop="taskNo" min-width="120" />
      <el-table-column :label="t('mes.productionTask.workOrderNo')" prop="workOrderNo" min-width="120" />
      <el-table-column :label="t('mes.productionTask.processName')" prop="processName" min-width="100" />
      <el-table-column :label="t('mes.productionTask.quantity')" prop="quantity" width="100" />
      <el-table-column :label="t('mes.productionTask.completedQuantity')" prop="completedQuantity" width="130" />
      <el-table-column :label="t('mes.productionTask.status')" prop="status" width="120">
        <template #default="{ row }">
          <el-tag :type="getStatusType(row.status)">
            {{ t(`mes.productionTask.status${row.status}`) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('mes.productionTask.operatorName')" prop="operatorName" width="100" />
      <el-table-column :label="t('mes.productionTask.startTime')" prop="startTime" width="160">
        <template #default="{ row }">
          {{ row.startTime ? formatDateTime(row.startTime) : '-' }}
        </template>
      </el-table-column>
      <el-table-column :label="t('common.operations')" width="200" fixed="right">
        <template #default="{ row }">
          <el-button type="primary" link @click="handleView(row)">
            {{ t('common.detail') }}
          </el-button>
          <el-button type="primary" link @click="handleUpdate(row)">
            {{ t('common.edit') }}
          </el-button>
          <el-button type="danger" link @click="handleDelete(row)">
            {{ t('common.delete') }}
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="listQuery.page"
      v-model:page-size="listQuery.pageSize"
      :total="total"
      :page-sizes="[10, 20, 50, 100]"
      layout="total, sizes, prev, pager, next, jumper"
      @size-change="getList"
      @current-change="getList"
    />
  </div>
</template>

<script setup lang="ts">
import { Search, Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { 
  getTaskListAPI, 
  deleteTaskAPI,
  type ProductionTask,
  type TaskStatus 
} from '@/apis/productionTask'

const { t } = useI18n()
const router = useRouter()

const list = ref<ProductionTask[]>([])
const listLoading = ref(false)
const total = ref(0)
const listQuery = ref({
  taskNo: '',
  status: '' as TaskStatus | '',
  page: 1,
  pageSize: 10,
})

const statusOptions = [
  { value: 'PENDING', label: 'mes.productionTask.statusPENDING' },
  { value: 'IN_PROGRESS', label: 'mes.productionTask.statusIN_PROGRESS' },
  { value: 'QUALITY_CHECK', label: 'mes.productionTask.statusQUALITY_CHECK' },
  { value: 'COMPLETED', label: 'mes.productionTask.statusCOMPLETED' },
  { value: 'CANCELLED', label: 'mes.productionTask.statusCANCELLED' },
  { value: 'ON_HOLD', label: 'mes.productionTask.statusON_HOLD' },
  { value: 'SCRAPPED', label: 'mes.productionTask.statusSCRAPPED' },
]

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    PENDING: 'info',
    IN_PROGRESS: 'warning',
    QUALITY_CHECK: 'warning',
    COMPLETED: 'success',
    CANCELLED: 'info',
    ON_HOLD: 'warning',
    SCRAPPED: 'danger',
  }
  return typeMap[status] || 'info'
}

const formatDateTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return date.toLocaleString()
}

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, unknown> = {}
    if (listQuery.value.taskNo) params.taskNo = listQuery.value.taskNo
    if (listQuery.value.status) params.status = listQuery.value.status
    
    const response = await getTaskListAPI(params)
    list.value = response.data || []
    total.value = list.value.length
  } catch (error) {
    ElMessage.error(t('mes.productionTask.loadFailed'))
  } finally {
    listLoading.value = false
  }
}

const handleFilter = () => {
  listQuery.value.page = 1
  getList()
}

const handleCreate = () => {
  router.push({ name: 'productionTaskForm' })
}

const handleUpdate = (row: ProductionTask) => {
  router.push({ name: 'productionTaskForm', query: { id: row.id } })
}

const handleView = (row: ProductionTask) => {
  router.push({ name: 'productionTaskDetail', query: { id: row.id } })
}

const handleDelete = async (row: ProductionTask) => {
  try {
    await ElMessageBox.confirm(
      t('mes.productionTask.confirmDelete'),
      t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' }
    )
    await deleteTaskAPI(row.id!)
    ElMessage.success(t('mes.productionTask.deleteSuccess'))
    getList()
  } catch {
    // user cancel
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.filter-container {
  margin-bottom: 20px;
}
.filter-item {
  margin-right: 10px;
}
</style>
