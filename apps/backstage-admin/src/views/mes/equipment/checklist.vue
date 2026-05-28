<template>
  <div class="checklist-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.equipment.checklist.equipment')">
          <el-select v-model="queryParams.equipmentId" :placeholder="t('mes.equipment.checklist.selectEquipment')" clearable filterable>
            <el-option v-for="eq in equipmentList" :key="eq.id" :label="`${eq.equipmentCode} - ${eq.equipmentName}`" :value="eq.id" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.checkType')">
          <el-select v-model="queryParams.checkType" :placeholder="t('mes.equipment.checklist.selectType')" clearable>
            <el-option :label="t('mes.equipment.checklist.typeDaily')" value="DAILY" />
            <el-option :label="t('mes.equipment.checklist.typeWeekly')" value="WEEKLY" />
            <el-option :label="t('mes.equipment.checklist.typeMonthly')" value="MONTHLY" />
            <el-option :label="t('mes.equipment.checklist.typeQuarterly')" value="QUARTERLY" />
            <el-option :label="t('mes.equipment.checklist.typeAnnual')" value="ANNUAL" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.equipment.checklist.selectStatus')" clearable>
            <el-option :label="t('mes.equipment.checklist.statusPending')" value="PENDING" />
            <el-option :label="t('mes.equipment.checklist.statusCompleted')" value="COMPLETED" />
            <el-option :label="t('mes.equipment.checklist.statusAbnormal')" value="ABNORMAL" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.dateRange')">
          <el-date-picker
            v-model="dateRange"
            type="daterange"
            range-separator="-"
            :start-placeholder="t('mes.equipment.checklist.startDate')"
            :end-placeholder="t('mes.equipment.checklist.endDate')"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
          <el-button @click="resetQuery">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">{{ t('mes.equipment.checklist.add') }}</el-button>
      </div>

      <el-table :data="checklistList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.equipment.checklist.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.equipment.checklist.checkCode')" prop="checkCode" width="150" />
        <el-table-column :label="t('mes.equipment.equipmentCode')" prop="equipmentCode" width="120" />
        <el-table-column :label="t('mes.equipment.equipmentName')" prop="equipmentName" width="150" />
        <el-table-column :label="t('mes.equipment.checklist.checkType')" prop="checkType" width="100">
          <template #default="{ row }">
            <el-tag :type="getTypeTag(row.checkType)">{{ getTypeText(row.checkType) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.equipment.checklist.checkTime')" prop="checkTime" width="160" />
        <el-table-column :label="t('mes.equipment.checklist.checker')" prop="checkerName" width="100" />
        <el-table-column :label="t('mes.equipment.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTag(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.equipment.checklist.result')" prop="result" width="200" show-overflow-tooltip />
        <el-table-column :label="t('mes.equipment.checklist.abnormalItems')" prop="abnormalItems" width="150" show-overflow-tooltip />
        <el-table-column :label="t('common.operation')" fixed="right" width="180">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleComplete(row)" v-if="row.status === 'PENDING'">{{ t('mes.equipment.checklist.complete') }}</el-button>
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
        <el-form-item :label="t('mes.equipment.checklist.equipment')" prop="equipmentId">
          <el-select v-model="form.equipmentId" :placeholder="t('mes.equipment.checklist.selectEquipment')" filterable>
            <el-option v-for="eq in equipmentList" :key="eq.id" :label="`${eq.equipmentCode} - ${eq.equipmentName}`" :value="eq.id" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.checkCode')" prop="checkCode">
          <el-input v-model="form.checkCode" :placeholder="t('mes.equipment.checklist.checkCodePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.checkType')" prop="checkType">
          <el-select v-model="form.checkType" :placeholder="t('mes.equipment.checklist.selectType')">
            <el-option :label="t('mes.equipment.checklist.typeDaily')" value="DAILY" />
            <el-option :label="t('mes.equipment.checklist.typeWeekly')" value="WEEKLY" />
            <el-option :label="t('mes.equipment.checklist.typeMonthly')" value="MONTHLY" />
            <el-option :label="t('mes.equipment.checklist.typeQuarterly')" value="QUARTERLY" />
            <el-option :label="t('mes.equipment.checklist.typeAnnual')" value="ANNUAL" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.checkTime')" prop="checkTime">
          <el-date-picker v-model="form.checkTime" type="datetime" :placeholder="t('mes.equipment.checklist.selectDateTime')" value-format="YYYY-MM-DD HH:mm:ss" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.checker')">
          <el-input v-model="form.checkerName" :placeholder="t('mes.equipment.checklist.checkerPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.remarks')">
          <el-input v-model="form.remarks" type="textarea" :rows="2" :placeholder="t('mes.equipment.checklist.remarksPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="submitForm" :loading="submitLoading">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>

    <!-- Complete Checklist Dialog -->
    <el-dialog v-model="completeDialogVisible" :title="t('mes.equipment.checklist.completeTitle')" width="500px">
      <el-form :model="completeForm" label-width="120px">
        <el-form-item :label="t('mes.equipment.checklist.result')">
          <el-input v-model="completeForm.result" type="textarea" :rows="3" :placeholder="t('mes.equipment.checklist.resultPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.abnormalItems')">
          <el-input v-model="completeForm.abnormalItems" type="textarea" :rows="2" :placeholder="t('mes.equipment.checklist.abnormalItemsPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.equipment.checklist.remarks')">
          <el-input v-model="completeForm.remarks" type="textarea" :rows="2" :placeholder="t('mes.equipment.checklist.remarksPlaceholder')" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="completeDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="submitComplete" :loading="completeLoading">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox, type FormInstance } from 'element-plus'
import type { EquipmentChecklist, CreateChecklistRequest } from '@/apis/equipment'
import { getEquipmentListAPI } from '@/apis/equipment'
import { 
  getChecklistListAPI, 
  createChecklistAPI, 
  updateChecklistAPI, 
  deleteChecklistAPI,
  completeChecklistAPI 
} from '@/apis/equipment'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const submitLoading = ref(false)
const completeLoading = ref(false)
const checklistList = ref<EquipmentChecklist[]>([])
const equipmentList = ref<any[]>([])
const total = ref(0)

const queryParams = reactive({
  equipmentId: undefined as number | undefined,
  checkType: '',
  status: '',
  startDate: '',
  endDate: '',
  pageNum: 1,
  pageSize: 10
})

const dateRange = ref<[string, string] | null>(null)

const dialogVisible = ref(false)
const dialogTitle = ref('')
const formRef = ref<FormInstance>()
const form = reactive<CreateChecklistRequest>({
  equipmentId: 0,
  checkCode: '',
  checkType: 'DAILY',
  checkTime: '',
  checkerId: '',
  checkerName: '',
  status: 'PENDING',
  result: '',
  abnormalItems: '',
  remarks: ''
})

const rules = {
  equipmentId: [{ required: true, message: t('mes.equipment.checklist.selectEquipment'), trigger: 'change' }],
  checkCode: [{ required: true, message: t('mes.equipment.checklist.checkCodePlaceholder'), trigger: 'blur' }],
  checkType: [{ required: true, message: t('mes.equipment.checklist.selectType'), trigger: 'change' }],
  checkTime: [{ required: true, message: t('mes.equipment.checklist.selectDateTime'), trigger: 'change' }]
}

const completeDialogVisible = ref(false)
const completeForm = reactive({
  result: '',
  abnormalItems: '',
  remarks: '',
  checklistId: 0
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
    ElMessage.error(t('mes.equipment.checklist.fetchEquipmentFailed'))
  }
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getChecklistListAPI({
      equipmentId: queryParams.equipmentId,
      checkType: queryParams.checkType || undefined,
      status: queryParams.status || undefined,
      startDate: dateRange.value?.[0],
      endDate: dateRange.value?.[1]
    })
    checklistList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.equipment.checklist.fetchFailed'))
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
  queryParams.checkType = ''
  queryParams.status = ''
  dateRange.value = null
  handleQuery()
}

const handleAdd = () => {
  dialogTitle.value = t('mes.equipment.checklist.add')
  Object.assign(form, {
    equipmentId: 0,
    checkCode: '',
    checkType: 'DAILY',
    checkTime: '',
    checkerId: '',
    checkerName: '',
    status: 'PENDING',
    result: '',
    abnormalItems: '',
    remarks: ''
  })
  dialogVisible.value = true
}

const handleEdit = (row: EquipmentChecklist) => {
  dialogTitle.value = t('mes.equipment.checklist.edit')
  Object.assign(form, {
    id: row.id,
    equipmentId: row.equipmentId,
    checkCode: row.checkCode,
    checkType: row.checkType,
    checkTime: row.checkTime,
    checkerId: row.checkerId,
    checkerName: row.checkerName,
    status: row.status,
    result: row.result,
    abnormalItems: row.abnormalItems,
    remarks: row.remarks
  })
  dialogVisible.value = true
}

const handleView = (row: EquipmentChecklist) => {
  router.push({ path: '/mes/equipment/checklist-detail', query: { id: row.id } })
}

const handleDelete = (row: EquipmentChecklist) => {
  ElMessageBox.confirm(t('mes.equipment.checklist.confirmDelete'), t('common.warning'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  }).then(async () => {
    try {
      await deleteChecklistAPI(row.id!)
      ElMessage.success(t('common.deleteSuccess'))
      getList()
    } catch (error) {
      ElMessage.error(t('common.deleteFailed'))
    }
  }).catch(() => {
    // ignore
  })
}

const handleComplete = (row: EquipmentChecklist) => {
  completeForm.result = ''
  completeForm.abnormalItems = ''
  completeForm.remarks = ''
  completeForm.checklistId = row.id!
  completeDialogVisible.value = true
}

const submitForm = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid) => {
    if (valid) {
      submitLoading.value = true
      try {
        if (form.id) {
          await updateChecklistAPI(form.id, form)
          ElMessage.success(t('common.updateSuccess'))
        } else {
          await createChecklistAPI(form as any)
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

const submitComplete = async () => {
  completeLoading.value = true
  try {
    await completeChecklistAPI(completeForm.checklistId, {
      result: completeForm.result,
      abnormalItems: completeForm.abnormalItems,
      remarks: completeForm.remarks
    })
    ElMessage.success(t('mes.equipment.checklist.completeSuccess'))
    completeDialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error(t('common.operationFailed'))
  } finally {
    completeLoading.value = false
  }
}

const getTypeTag = (type: string) => {
  const map: Record<string, string> = {
    DAILY: 'success',
    WEEKLY: 'info',
    MONTHLY: 'warning',
    QUARTERLY: 'warning',
    ANNUAL: 'danger'
  }
  return map[type] || 'info'
}

const getTypeText = (type: string) => {
  const map: Record<string, string> = {
    DAILY: t('mes.equipment.checklist.typeDaily'),
    WEEKLY: t('mes.equipment.checklist.typeWeekly'),
    MONTHLY: t('mes.equipment.checklist.typeMonthly'),
    QUARTERLY: t('mes.equipment.checklist.typeQuarterly'),
    ANNUAL: t('mes.equipment.checklist.typeAnnual')
  }
  return map[type] || type
}

const getStatusTag = (status: string) => {
  const map: Record<string, string> = {
    PENDING: 'warning',
    COMPLETED: 'success',
    ABNORMAL: 'danger'
  }
  return map[status] || 'info'
}

const getStatusText = (status: string) => {
  const map: Record<string, string> = {
    PENDING: t('mes.equipment.checklist.statusPending'),
    COMPLETED: t('mes.equipment.checklist.statusCompleted'),
    ABNORMAL: t('mes.equipment.checklist.statusAbnormal')
  }
  return map[status] || status
}
</script>

<style scoped>
.checklist-container {
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