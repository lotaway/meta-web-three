<template>
  <div class="app-container">
    <div class="filter-container">
      <el-input
        v-model="listQuery.workOrderNo"
        :placeholder="t('mes.workOrder.workOrderNoPlaceholder')"
        class="filter-item"
        style="width: 200px"
        @keyup.enter="handleFilter"
      />
      <el-select
        v-model="listQuery.status"
        :placeholder="t('mes.workOrder.statusPlaceholder')"
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
        {{ t('mes.workOrder.add') }}
      </el-button>
    </div>

    <el-table v-loading="listLoading" :data="list" border style="width: 100%">
      <el-table-column :label="t('common.id')" prop="id" width="80" />
      <el-table-column :label="t('mes.workOrder.workOrderNo')" prop="workOrderNo" min-width="120" />
      <el-table-column :label="t('mes.workOrder.productCode')" prop="productCode" min-width="100" />
      <el-table-column :label="t('mes.workOrder.productName')" prop="productName" min-width="120" />
      <el-table-column :label="t('mes.workOrder.quantity')" prop="quantity" width="100" />
      <el-table-column :label="t('mes.workOrder.completedQuantity')" prop="completedQuantity" width="120" />
      <el-table-column :label="t('mes.workOrder.status')" prop="status" width="120">
        <template #default="{ row }">
          <el-tag :type="getStatusType(row.status)">
            {{ t(`mes.workOrder.status${row.status}`) }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column :label="t('mes.workOrder.typeCode')" prop="typeCode" width="100">
        <template #default="{ row }">
          {{ row.typeCode ? t(`mes.workOrder.typeCode${row.typeCode}`) : '-' }}
        </template>
      </el-table-column>
      <el-table-column :label="t('mes.workOrder.priority')" prop="priority" width="100">
        <template #default="{ row }">
          {{ row.priority ? t(`mes.workOrder.priority${row.priority}`) : '-' }}
        </template>
      </el-table-column>
      <el-table-column :label="t('mes.workOrder.completionRate')" prop="completionRate" width="100">
        <template #default="{ row }">
          {{ row.completionRate ? `${row.completionRate}%` : '0%' }}
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
  getWorkOrderListAPI,
  deleteWorkOrderAPI,
  type WorkOrder,
  type WorkOrderStatus
} from '@/apis/workOrder'

const { t } = useI18n()
const router = useRouter()

const list = ref<WorkOrder[]>([])
const listLoading = ref(false)
const total = ref(0)
const listQuery = ref({
  workOrderNo: '',
  status: '' as WorkOrderStatus | '',
  page: 1,
  pageSize: 10,
})

const statusOptions = [
  { value: 'DRAFT', label: 'mes.workOrder.statusDRAFT' },
  { value: 'RELEASED', label: 'mes.workOrder.statusRELEASED' },
  { value: 'IN_PROGRESS', label: 'mes.workOrder.statusIN_PROGRESS' },
  { value: 'PAUSED', label: 'mes.workOrder.statusPAUSED' },
  { value: 'COMPLETED', label: 'mes.workOrder.statusCOMPLETED' },
  { value: 'CANCELLED', label: 'mes.workOrder.statusCANCELLED' },
]

const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    DRAFT: 'info',
    RELEASED: 'warning',
    IN_PROGRESS: 'primary',
    PAUSED: 'warning',
    COMPLETED: 'success',
    CANCELLED: 'info',
  }
  return typeMap[status] || 'info'
}

const getList = async () => {
  listLoading.value = true
  try {
    const params: Record<string, unknown> = {}
    if (listQuery.value.workOrderNo) params.workOrderNo = listQuery.value.workOrderNo
    if (listQuery.value.status) params.status = listQuery.value.status

    const response = await getWorkOrderListAPI(params)
    list.value = response.data || []
    total.value = list.value.length
  } catch (error) {
    ElMessage.error(t('mes.workOrder.loadFailed'))
  } finally {
    listLoading.value = false
  }
}

const handleFilter = () => {
  listQuery.value.page = 1
  getList()
}

const handleCreate = () => {
  router.push({ name: 'workOrderForm' })
}

const handleUpdate = (row: WorkOrder) => {
  router.push({ name: 'workOrderForm', query: { id: row.id } })
}

const handleView = (row: WorkOrder) => {
  router.push({ name: 'workOrderDetail', query: { id: row.id } })
}

const handleDelete = async (row: WorkOrder) => {
  try {
    await ElMessageBox.confirm(
      t('mes.workOrder.confirmDelete'),
      t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' }
    )
    await deleteWorkOrderAPI(row.id!)
    ElMessage.success(t('mes.workOrder.deleteSuccess'))
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