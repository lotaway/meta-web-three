<template>
  <div class="process-route-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.processRoute.edit') : t('mes.processRoute.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.processRoute.routeCode')" prop="routeCode">
              <el-input v-model="form.routeCode" :placeholder="t('mes.processRoute.routeCodePlaceholder')" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.processRoute.routeName')" prop="routeName">
              <el-input v-model="form.routeName" :placeholder="t('mes.processRoute.routeNamePlaceholder')" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.processRoute.productCode')" prop="productCode">
              <el-input v-model="form.productCode" :placeholder="t('mes.processRoute.productCodePlaceholder')" />
            </el-form-item>
          </el-col>
          <el-col :span="12" v-if="isEdit">
            <el-form-item :label="t('mes.processRoute.version')">
              <el-input :value="form.version" disabled />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">{{ t('mes.processRoute.steps') }}</el-divider>

        <div class="steps-container">
          <el-table :data="form.steps" border stripe>
            <el-table-column :label="t('mes.processRoute.stepNo')" prop="stepNo" width="80" align="center">
              <template #default="{ $index }">
                {{ $index + 1 }}
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.processRoute.processCode')" prop="processCode" min-width="120">
              <template #default="{ row }">
                <el-input v-model="row.processCode" :placeholder="t('mes.processRoute.processCodePlaceholder')" size="small" />
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.processRoute.processName')" prop="processName" min-width="120">
              <template #default="{ row }">
                <el-input v-model="row.processName" :placeholder="t('mes.processRoute.processNamePlaceholder')" size="small" />
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.processRoute.workstationId')" prop="workstationId" width="120">
              <template #default="{ row }">
                <el-input v-model="row.workstationId" :placeholder="t('mes.processRoute.workstationIdPlaceholder')" size="small" />
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.processRoute.standardTime')" prop="standardTime" width="140">
              <template #default="{ row }">
                <el-input-number v-model="row.standardTime" :min="0" :step="1" size="small" controls-position="right" />
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.processRoute.qualityCheckpoint')" prop="qualityCheckpoint" width="120">
              <template #default="{ row }">
                <el-switch v-model="row.qualityCheckpoint" active-value="YES" inactive-value="NO" />
              </template>
            </el-table-column>
            <el-table-column :label="t('mes.processRoute.predecessorStep')" prop="predecessorStepNo" width="100">
              <template #default="{ row, $index }">
                <el-select v-model="row.predecessorStepNo" :placeholder="t('mes.processRoute.selectPredecessor')" clearable size="small">
                  <el-option v-for="(step, idx) in getAvailablePredecessors($index)" :key="idx" :label="step.processName || t('mes.processRoute.step') + (idx+1)" :value="step.stepNo" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column :label="t('common.operation')" width="80" align="center">
              <template #default="{ $index }">
                <el-button link type="danger" size="small" @click="removeStep($index)">{{ t('mes.processRoute.removeStep') }}</el-button>
              </template>
            </el-table-column>
          </el-table>

          <el-button type="primary" plain size="small" @click="addStep" style="margin-top: 10px">
            <el-icon><Plus /></el-icon>
            {{ t('mes.processRoute.addStep') }}
          </el-button>
        </div>

        <el-divider />

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('common.submit') }}</el-button>
          <el-button @click="handleCancel">{{ t('common.cancelText') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import type { ProcessRoute, ProcessStep, CreateProcessRouteRequest, UpdateProcessRouteRequest } from '@/apis/processRoute'
import { 
  getProcessRouteByIdAPI, 
  createProcessRouteAPI, 
  updateProcessRouteAPI,
  validateProcessRouteAPI 
} from '@/apis/processRoute'

const { t } = useI18n()

const route = useRoute()
const router = useRouter()
const formRef = ref<FormInstance>()
const submitting = ref(false)

const isEdit = computed(() => !!route.query.id)
const routeId = computed(() => Number(route.query.id) || 0)

const form = reactive<{
  routeCode: string
  routeName: string
  productCode: string
  version?: number
  status?: string
  steps: ProcessStep[]
}>({
  routeCode: '',
  routeName: '',
  productCode: '',
  version: undefined,
  status: undefined,
  steps: []
})

const rules: FormRules = {
  routeCode: [{ required: true, message: t('mes.processRoute.routeCodeRequired'), trigger: 'blur' }],
  routeName: [{ required: true, message: t('mes.processRoute.routeNameRequired'), trigger: 'blur' }],
  productCode: [{ required: true, message: t('mes.processRoute.productCodeRequired'), trigger: 'blur' }],
}

const addStep = () => {
  const stepNo = form.steps.length + 1
  form.steps.push({
    stepNo,
    processCode: '',
    processName: '',
    workstationId: undefined,
    standardTime: undefined,
    qualityCheckpoint: 'NO',
    predecessorStepNo: undefined,
    successorStepNo: undefined
  })
}

const removeStep = (index: number) => {
  form.steps.splice(index, 1)
  form.steps.forEach((step, idx) => {
    step.stepNo = idx + 1
  })
}

const getAvailablePredecessors = (currentIndex: number) => {
  return form.steps.filter((_, idx) => idx < currentIndex)
}

const loadData = async () => {
  if (!isEdit.value) return
  
  try {
    const res = await getProcessRouteByIdAPI(routeId.value)
    const data = res.data
    if (data) {
      form.routeCode = data.routeCode
      form.routeName = data.routeName
      form.productCode = data.productCode
      form.version = data.version
      form.status = data.status
      form.steps = data.steps ? [...data.steps] : []
    }
  } catch (error: unknown) {
    ElMessage.error(t('mes.processRoute.loadFailed'))
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    if (form.steps.length === 0) {
      ElMessage.warning(t('mes.processRoute.addAtLeastOneStep'))
      return
    }

    submitting.value = true
    try {
      if (isEdit.value) {
        const data: UpdateProcessRouteRequest = {
          routeName: form.routeName,
          productCode: form.productCode,
          steps: form.steps
        }
        await updateProcessRouteAPI(routeId.value, data)
        ElMessage.success(t('mes.processRoute.updateSuccess'))
      } else {
        const data: CreateProcessRouteRequest = {
          routeCode: form.routeCode,
          routeName: form.routeName,
          productCode: form.productCode,
          steps: form.steps
        }
        await createProcessRouteAPI(data)
        ElMessage.success(t('mes.processRoute.createSuccess'))
      }

      router.push('/mes/process-route')
    } catch (error: unknown) {
      ElMessage.error(t('mes.processRoute.submitFailed') + (error as Error).message)
    } finally {
      submitting.value = false
    }
  })
}

const handleCancel = () => {
  router.push('/mes/process-route')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.process-route-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}

.steps-container {
  margin: 10px 0;
}
</style>