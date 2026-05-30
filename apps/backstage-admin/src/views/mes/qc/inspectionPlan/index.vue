<template>
  <div class="inspection-plan-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.inspectionPlan.planCode')">
          <el-input v-model="queryParams.planCode" :placeholder="t('mes.qc.inspectionPlan.planCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionPlan.planName')">
          <el-input v-model="queryParams.planName" :placeholder="t('mes.qc.inspectionPlan.planNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionPlan.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.qc.inspectionPlan.statusPlaceholder')" clearable>
            <el-option :label="t('mes.qc.inspectionPlan.statusDraft')" value="DRAFT" />
            <el-option :label="t('mes.qc.inspectionPlan.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.inspectionPlan.statusSuspended')" value="SUSPENDED" />
            <el-option :label="t('mes.qc.inspectionPlan.statusArchived')" value="ARCHIVED" />
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
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.inspectionPlan.add') }}</el-button>
      </div>

      <el-table :data="planList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.inspectionPlan.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.inspectionPlan.planCode')" prop="planCode" width="150" />
        <el-table-column :label="t('mes.qc.inspectionPlan.planName')" prop="planName" width="180" />
        <el-table-column :label="t('mes.qc.inspectionPlan.inspectionType')" prop="inspectionType" width="120" />
        <el-table-column :label="t('mes.qc.inspectionPlan.applicableProducts')" prop="applicableProducts" width="150" />
        <el-table-column :label="t('mes.qc.inspectionPlan.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.inspectionPlan.version')" prop="version" width="80" />
        <el-table-column :label="t('mes.qc.inspectionPlan.effectiveDate')" prop="effectiveDate" width="120" />
        <el-table-column :label="t('mes.qc.inspectionPlan.expiryDate')" prop="expiryDate" width="120" />
        <el-table-column :label="t('mes.qc.inspectionPlan.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operation')" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleActivate(row)" v-if="row.status === 'DRAFT'">{{ t('mes.qc.inspectionPlan.activate') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDeactivate(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.qc.inspectionPlan.deactivate') }}</el-button>
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
import type { QcInspectionPlan, PlanStatus } from '@/apis/qc'
import {
  getInspectionPlanListAPI,
  deleteInspectionPlanAPI,
  activateInspectionPlanAPI,
  deactivateInspectionPlanAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const planList = ref<QcInspectionPlan[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  planCode: '',
  planName: '',
  status: '' as PlanStatus | ''
})

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    SUSPENDED: 'warning',
    ARCHIVED: 'info'
  }
  return typeMap[status] || 'info'
}

const getStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    DRAFT: t('mes.qc.inspectionPlan.statusDraft'),
    ACTIVE: t('mes.qc.inspectionPlan.statusActive'),
    SUSPENDED: t('mes.qc.inspectionPlan.statusSuspended'),
    ARCHIVED: t('mes.qc.inspectionPlan.statusArchived')
  }
  return textMap[status] || status
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getInspectionPlanListAPI()
    planList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionPlan.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.planCode = ''
  queryParams.planName = ''
  queryParams.status = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/inspectionPlan/form' })
}

const handleView = (row: QcInspectionPlan) => {
  router.push({ path: '/mes/inspectionPlan/detail', query: { id: row.id } })
}

const handleEdit = (row: QcInspectionPlan) => {
  router.push({ path: '/mes/inspectionPlan/form', query: { id: row.id } })
}

const handleActivate = async (row: QcInspectionPlan) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionPlan.confirmActivate'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await activateInspectionPlanAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionPlan.activateSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionPlan.activateFailed'))
    }
  }
}

const handleDeactivate = async (row: QcInspectionPlan) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionPlan.confirmDeactivate'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deactivateInspectionPlanAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionPlan.deactivateSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionPlan.deactivateFailed'))
    }
  }
}

const handleDelete = async (row: QcInspectionPlan) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionPlan.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteInspectionPlanAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionPlan.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionPlan.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.inspection-plan-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>