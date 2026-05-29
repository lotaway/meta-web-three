<template>
  <div class="inspection-plan-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.inspectionPlan.edit') : t('mes.qc.inspectionPlan.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="150px">
        <el-form-item :label="t('mes.qc.inspectionPlan.planCode')" prop="planCode">
          <el-input v-model="formData.planCode" :disabled="isEdit" :placeholder="t('mes.qc.inspectionPlan.planCodePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.planName')" prop="planName">
          <el-input v-model="formData.planName" :placeholder="t('mes.qc.inspectionPlan.planNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.inspectionType')" prop="inspectionType">
          <el-input v-model="formData.inspectionType" :placeholder="t('mes.qc.inspectionPlan.inspectionTypePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.applicableProducts')" prop="applicableProducts">
          <el-input v-model="formData.applicableProducts" :placeholder="t('mes.qc.inspectionPlan.applicableProductsPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.status')" prop="status">
          <el-select v-model="formData.status" :placeholder="t('mes.qc.inspectionPlan.statusPlaceholder')">
            <el-option :label="t('mes.qc.inspectionPlan.statusDraft')" value="DRAFT" />
            <el-option :label="t('mes.qc.inspectionPlan.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.inspectionPlan.statusSuspended')" value="SUSPENDED" />
            <el-option :label="t('mes.qc.inspectionPlan.statusArchived')" value="ARCHIVED" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.effectiveDate')" prop="effectiveDate">
          <el-date-picker v-model="formData.effectiveDate" type="date" value-format="YYYY-MM-DD" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.expiryDate')" prop="expiryDate">
          <el-date-picker v-model="formData.expiryDate" type="date" value-format="YYYY-MM-DD" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.version')" prop="version">
          <el-input-number v-model="formData.version" :min="1" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionPlan.description')" prop="description">
          <el-input v-model="formData.description" type="textarea" :rows="3" :placeholder="t('mes.qc.inspectionPlan.descriptionPlaceholder')" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('mes.qc.inspectionPlan.save') }}</el-button>
          <el-button @click="handleCancel">{{ t('mes.qc.inspectionPlan.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { FormInstance } from 'element-plus'
import type { QcInspectionPlan, PlanStatus } from '@/apis/qc'
import {
  getInspectionPlanByIdAPI,
  createInspectionPlanAPI,
  updateInspectionPlanAPI
} from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const formData = reactive<QcInspectionPlan>({
  planCode: '',
  planName: '',
  inspectionType: '',
  applicableProducts: '',
  status: 'DRAFT' as PlanStatus,
  effectiveDate: '',
  expiryDate: '',
  version: 1,
  description: '',
  planItems: []
})

const { t } = useI18n()

const rules = {
  planCode: [
    { required: true, message: t('mes.qc.inspectionPlan.planCodePlaceholder'), trigger: 'blur' }
  ],
  planName: [
    { required: true, message: t('mes.qc.inspectionPlan.planNamePlaceholder'), trigger: 'blur' }
  ],
  inspectionType: [
    { required: true, message: t('mes.qc.inspectionPlan.inspectionTypePlaceholder'), trigger: 'blur' }
  ]
}

const loadData = async () => {
  if (!route.query.id) return
  
  try {
    const res = await getInspectionPlanByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionPlan.loadFailed'))
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    submitting.value = true
    try {
      if (isEdit.value) {
        await updateInspectionPlanAPI(Number(route.query.id), formData)
        ElMessage.success(t('mes.qc.inspectionPlan.updateSuccess'))
      } else {
        await createInspectionPlanAPI(formData)
        ElMessage.success(t('mes.qc.inspectionPlan.createSuccess'))
      }
      router.push('/mes/inspectionPlan')
    } catch (error) {
      ElMessage.error(t('mes.qc.inspectionPlan.deleteFailed'))
    } finally {
      submitting.value = false
    }
  })
}

const handleCancel = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.inspection-plan-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}
</style>