<template>
  <div class="maintenance-plan-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.equipment.maintenancePlan.equipment')">
          <el-select v-model="queryParams.equipmentId" :placeholder="t('mes.equipment.maintenancePlan.selectEquipment')" clearable filterable>
            <el-option v-for="eq in equipmentList" :key="eq.id" :label="`${eq.equipmentCode} - ${eq.equipmentName}`" :value="eq.id" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.type')">
          <el-select v-model="queryParams.maintenanceType" :placeholder="t('mes.equipment.maintenancePlan.selectType')" clearable>
            <el-option :label="t('mes.equipment.maintenancePlan.typePreventive')" value="PREVENTIVE" />
            <el-option :label="t('mes.equipment.maintenancePlan.typePredictive')" value="PREDICTIVE" />
            <el-option :label="t('mes.equipment.maintenancePlan.typeCorrective')" value="CORRECTIVE" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.equipment.maintenancePlan.selectStatus')" clearable>
            <el-option :label="t('mes.equipment.maintenancePlan.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.equipment.maintenancePlan.statusInactive')" value="INACTIVE" />
            <el-option :label="t('mes.equipment.maintenancePlan.statusCompleted')" value="COMPLETED" />
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
        <el-button type="primary" @click="handleAdd">{{ t('mes.equipment.maintenancePlan.add') }}</el-button>
      </div>

      <el-table :data="planList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.equipment.maintenancePlan.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.equipment.maintenancePlan.planCode')" prop="planCode" width="150" />
        <el-table-column :label="t('mes.equipment.maintenancePlan.planName')" prop="planName" width="180" />
        <el-table-column :label="t('mes.equipment.equipmentCode')" prop="equipmentCode" width="120" />
        <el-table-column :label="t('mes.equipment.equipmentName')" prop="equipmentName" width="150" />
        <el-table-column :label="t('mes.equipment.maintenancePlan.type')" prop="maintenanceType" width="120">
          <template #default="{ row }">
            <el-tag :type="getTypeTag(row.maintenanceType)">{{ getTypeText(row.maintenanceType) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.equipment.maintenancePlan.interval')" width="120">
          <template #default="{ row }">
            {{ row.intervalDays }}{{ t('mes.equipment.maintenancePlan.days') }} / {{ row.intervalHours }}{{ t('mes.equipment.maintenancePlan.hours') }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.equipment.maintenancePlan.nextExecution')" prop="nextExecutionTime" width="160" />
        <el-table-column :label="t('mes.equipment.maintenancePlan.lastExecution')" prop="lastExecutionTime" width="160" />
        <el-table-column :label="t('mes.equipment.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTag(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('common.operation')" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleExecute(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.equipment.maintenancePlan.execute') }}</el-button>
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

    <!-- Add/Edit Dialog -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="140px">
        <el-form-item :label="t('mes.equipment.maintenancePlan.equipment')" prop="equipmentId">
          <el-select v-model="form.equipmentId" :placeholder="t('mes.equipment.maintenancePlan.selectEquipment')" filterable>
            <el-option v-for="eq in equipmentList" :key="eq.id" :label="`${eq.equipmentCode} - ${eq.equipmentName}`" :value="eq.id" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.planCode')" prop="planCode">
          <el-input v-model="form.planCode" :placeholder="t('mes.equipment.maintenancePlan.planCodePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.planName')" prop="planName">
          <el-input v-model="form.planName" :placeholder="t('mes.equipment.maintenancePlan.planNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.type')" prop="maintenanceType">
          <el-select v-model="form.maintenanceType" :placeholder="t('mes.equipment.maintenancePlan.selectType')">
            <el-option :label="t('mes.equipment.maintenancePlan.typePreventive')" value="PREVENTIVE" />
            <el-option :label="t('mes.equipment.maintenancePlan.typePredictive')" value="PREDICTIVE" />
            <el-option :label="t('mes.equipment.maintenancePlan.typeCorrective')" value="CORRECTIVE" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.intervalDays')" prop="intervalDays">
          <el-input-number v-model="form.intervalDays" :min="0" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.intervalHours')" prop="intervalHours">
          <el-input-number v-model="form.intervalHours" :min="0" :max="24" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.nextExecution')" prop="nextExecutionTime">
          <el-date-picker v-model="form.nextExecutionTime" type="datetime" :placeholder="t('mes.equipment.maintenancePlan.selectDateTime')" value-format="YYYY-MM-DD HH:mm:ss" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.duration')" prop="estimatedDuration">
          <el-input-number v-model="form.estimatedDuration" :min="1" /> {{ t('mes.equipment.maintenancePlan.minutes') }}
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.assignedTo')">
          <el-input v-model="form.assignedTo" :placeholder="t('mes.equipment.maintenancePlan.assignedToPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.checkItems')">
          <el-input v-model="form.checkItems" type="textarea" :rows="3" :placeholder="t('mes.equipment.maintenancePlan.checkItemsPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.maintenancePlan.description')">
          <el-input v-model="form.description" type="textarea" :rows="2" :placeholder="t('mes.equipment.maintenancePlan.descriptionPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="submitForm" :loading="submitLoading">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>

    <!-- Execute Maintenance Dialog -->
    <el-dialog v-model="executeDialogVisible" :title="t('mes.equipment.maintenancePlan.executeTitle')" width="400px">
      <el-form :model="executeForm" label-width="100px">
        <el-form-item :label="t('mes.equipment.maintenancePlan.result')">
          <el-input v-model="executeForm.result" type="textarea" :rows="3" :placeholder="t('mes.equipment.maintenancePlan.resultPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="executeDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="submitExecute" :loading="executeLoading">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox, type FormInstance } from 'element-plus'
import type { MaintenancePlan, CreateMaintenancePlanRequest } from '@/apis/equipment'
import { getEquipmentListAPI } from '@/apis/equipment'
import { 
  getMaintenancePlanListAPI, 
  createMaintenancePlanAPI, 
  updateMaintenancePlanAPI, 
  deleteMaintenancePlanAPI,
  executeMaintenancePlanAPI 
} from '@/apis/equipment'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const submitLoading = ref(false)
const executeLoading = ref(false)
const planList = ref<MaintenancePlan[]>([])
const equipmentList = ref<any[]>([])
const total = ref(0)

const queryParams = reactive({
  equipmentId: undefined as number | undefined,
  maintenanceType: '',
  status: '',
  pageNum: 1,
  pageSize: 10
})

const dialogVisible = ref(false)
const dialogTitle = ref('')
const formRef = ref<FormInstance>()
const form = reactive<CreateMaintenancePlanRequest>({
  equipmentId: 0,
  planCode: '',
  planName: '',
  maintenanceType: 'PREVENTIVE',
  intervalDays: 30,
  intervalHours: 0,
  nextExecutionTime: '',
  estimatedDuration: 60,
  assignedTo: '',
  description: '',
  checkItems: ''
})

const rules = {
  equipmentId: [{ required: true, message: t('mes.equipment.maintenancePlan.selectEquipment'), trigger: 'change' }],
  planCode: [{ required: true, message: t('mes.equipment.maintenancePlan.planCodePlaceholder'), trigger: 'blur' }],
  planName: [{ required: true, message: t('mes.equipment.maintenancePlan.planNamePlaceholder'), trigger: 'blur' }],
  maintenanceType: [{ required: true, message: t('mes.equipment.maintenancePlan.selectType'), trigger: 'change' }],
  nextExecutionTime: [{ required: true, message: t('mes.equipment.maintenancePlan.selectDateTime'), trigger: 'change' }]
}

const executeDialogVisible = ref(false)
const executeForm = reactive({
  result: '',
  planId: 0
})

onMounted(() => {
  getEquipmentList()
  getList()
})

const getEquipmentList = async () => {
  try {
    const res = await getEquipmentListAPI()
    equipmentList.value = res.data || []
  } catch (error) {
    ElMessage.error(t('mes.equipment.maintenancePlan.fetchEquipmentFailed'))
  }
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getMaintenancePlanListAPI({
      equipmentId: queryParams.equipmentId,
      maintenanceType: queryParams.maintenanceType || undefined,
      status: queryParams.status || undefined
    })
    planList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.equipment.maintenancePlan.fetchFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.equipmentId = undefined
  queryParams.maintenanceType = ''
  queryParams.status = ''
  handleQuery()
}

const handleAdd = () => {
  dialogTitle.value = t('mes.equipment.maintenancePlan.add')
  Object.assign(form, {
    equipmentId: 0,
    planCode: '',
    planName: '',
    maintenanceType: 'PREVENTIVE',
    intervalDays: 30,
    intervalHours: 0,
    nextExecutionTime: '',
    estimatedDuration: 60,
    assignedTo: '',
    description: '',
    checkItems: ''
  })
  dialogVisible.value = true
}

const handleEdit = (row: MaintenancePlan) => {
  dialogTitle.value = t('mes.equipment.maintenancePlan.edit')
  Object.assign(form, {
    id: row.id,
    equipmentId: row.equipmentId,
    planCode: row.planCode,
    planName: row.planName,
    maintenanceType: row.maintenanceType,
    intervalDays: row.intervalDays,
    intervalHours: row.intervalHours,
    nextExecutionTime: row.nextExecutionTime,
    estimatedDuration: row.estimatedDuration,
    assignedTo: row.assignedTo,
    description: row.description,
    checkItems: row.checkItems
  })
  dialogVisible.value = true
}

const handleView = (row: MaintenancePlan) => {
  router.push({ path: '/mes/equipment/maintenance-plan-detail', query: { id: row.id } })
}

const handleDelete = (row: MaintenancePlan) => {
  ElMessageBox.confirm(t('mes.equipment.maintenancePlan.confirmDelete'), t('common.warning'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  }).then(async () => {
    try {
      await deleteMaintenancePlanAPI(row.id!)
      ElMessage.success(t('common.deleteSuccess'))
      getList()
    } catch (error) {
      ElMessage.error(t('common.deleteFailed'))
    }
  }).catch(() => {
    // ignore
  })
}

const handleExecute = (row: MaintenancePlan) => {
  executeForm.result = ''
  executeForm.planId = row.id!
  executeDialogVisible.value = true
}

const submitForm = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid) => {
    if (valid) {
      submitLoading.value = true
      try {
        if (form.id) {
          await updateMaintenancePlanAPI(form.id, form)
          ElMessage.success(t('common.updateSuccess'))
        } else {
          await createMaintenancePlanAPI(form as any)
          ElMessage.success(t('common.addSuccess'))
        }
        dialogVisible.value = false
        getList()
      } catch (error) {
        ElMessage.error(t('common.operationFailed'))
      } finally {
        submitLoading.value = false
      }
    }
  })
}

const submitExecute = async () => {
  executeLoading.value = true
  try {
    await executeMaintenancePlanAPI(executeForm.planId, executeForm.result)
    ElMessage.success(t('mes.equipment.maintenancePlan.executeSuccess'))
    executeDialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error(t('common.operationFailed'))
  } finally {
    executeLoading.value = false
  }
}

const getTypeTag = (type: string) => {
  const map: Record<string, string> = {
    PREVENTIVE: 'success',
    PREDICTIVE: 'warning',
    CORRECTIVE: 'danger'
  }
  return map[type] || 'info'
}

const getTypeText = (type: string) => {
  const map: Record<string, string> = {
    PREVENTIVE: t('mes.equipment.maintenancePlan.typePreventive'),
    PREDICTIVE: t('mes.equipment.maintenancePlan.typePredictive'),
    CORRECTIVE: t('mes.equipment.maintenancePlan.typeCorrective')
  }
  return map[type] || type
}

const getStatusTag = (status: string) => {
  const map: Record<string, string> = {
    ACTIVE: 'success',
    INACTIVE: 'info',
    COMPLETED: 'warning'
  }
  return map[status] || 'info'
}

const getStatusText = (status: string) => {
  const map: Record<string, string> = {
    ACTIVE: t('mes.equipment.maintenancePlan.statusActive'),
    INACTIVE: t('mes.equipment.maintenancePlan.statusInactive'),
    COMPLETED: t('mes.equipment.maintenancePlan.statusCompleted')
  }
  return map[status] || status
}
</script>

<style scoped>
.maintenance-plan-container {
  padding: 20px;
}
.filter-card {
  margin-bottom: 20px;
}
.filter-form {
  display: flex;
  flex-wrap: wrap;
}
.toolbar {
  margin-bottom: 15px;
}
</style>