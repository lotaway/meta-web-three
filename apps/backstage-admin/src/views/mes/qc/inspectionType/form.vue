<template>
  <div class="inspection-type-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.inspectionType.edit') : t('mes.qc.inspectionType.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="150px">
        <el-form-item :label="t('mes.qc.inspectionType.typeCode')" prop="typeCode">
          <el-input v-model="formData.typeCode" :disabled="isEdit" :placeholder="t('mes.qc.inspectionType.typeCodePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionType.typeName')" prop="typeName">
          <el-input v-model="formData.typeName" :placeholder="t('mes.qc.inspectionType.typeNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionType.category')" prop="category">
          <el-select v-model="formData.category" :placeholder="t('mes.qc.inspectionType.categoryPlaceholder')">
            <el-option :label="t('mes.qc.inspectionType.categoryIncoming')" value="INCOMING" />
            <el-option :label="t('mes.qc.inspectionType.categoryProcess')" value="PROCESS" />
            <el-option :label="t('mes.qc.inspectionType.categoryFinal')" value="FINAL" />
            <el-option :label="t('mes.qc.inspectionType.categoryOutgoing')" value="OUTGOING" />
            <el-option :label="t('mes.qc.inspectionType.categoryCustom')" value="CUSTOM" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.equipment.description')" prop="description">
          <el-input v-model="formData.description" type="textarea" :rows="3" :placeholder="t('mes.equipment.descriptionPlaceholder')" />
        </el-form-item>

        <el-form-item label="Applicable Products" prop="applicableProducts">
          <el-input v-model="formData.applicableProducts" placeholder="Enter applicable products, separated by commas" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionType.defaultSamplingPlan')" prop="defaultSamplingPlan">
          <el-input v-model="formData.defaultSamplingPlan" placeholder="Example: GB/T 2828.1" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionType.defaultAql')" prop="defaultAql">
          <el-input v-model="formData.defaultAql" placeholder="例如: 0.1, 0.15, 0.25" />
        </el-form-item>

        <el-form-item label="Default Inspection Timeout (Hours)" prop="defaultTimeoutHours">
          <el-input-number v-model="formData.defaultTimeoutHours" :min="1" :max="168" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionType.requireCertificate')" prop="requireCertificate">
          <el-switch v-model="formData.requireCertificate" />
        </el-form-item>

        <el-form-item label="Require Test Report" prop="requireTestReport">
          <el-switch v-model="formData.requireTestReport" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.inspectionType.sortOrder')" prop="sortOrder">
          <el-input-number v-model="formData.sortOrder" :min="0" :max="9999" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('common.save') }}</el-button>
          <el-button @click="handleCancel">{{ t('common.cancel') }}</el-button>
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
import type { QcInspectionType, InspectionCategory } from '@/apis/qc'
import {
  getInspectionTypeByIdAPI,
  createInspectionTypeAPI,
  updateInspectionTypeAPI
} from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const formData = reactive<QcInspectionType>({
  typeCode: '',
  typeName: '',
  category: 'INCOMING' as InspectionCategory,
  description: '',
  applicableProducts: '',
  defaultSamplingPlan: '',
  defaultAql: '',
  defaultTimeoutHours: 24,
  requireCertificate: false,
  requireTestReport: false,
  sortOrder: 0,
  status: 'INACTIVE'
})

const { t } = useI18n()

const rules = {
  typeCode: [
    { required: true, message: t('mes.qc.inspectionType.typeCodePlaceholder'), trigger: 'blur' }
  ],
  typeName: [
    { required: true, message: t('mes.qc.inspectionType.typeNamePlaceholder'), trigger: 'blur' }
  ],
  category: [
    { required: true, message: t('mes.qc.inspectionType.categoryPlaceholder'), trigger: 'change' }
  ]
}

const loadData = async () => {
  if (!route.query.id) return
  
  try {
    const res = await getInspectionTypeByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.processRoute.loadFailed'))
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    submitting.value = true
    try {
      if (isEdit.value) {
        await updateInspectionTypeAPI(Number(route.query.id), formData)
        ElMessage.success(t('mes.qc.inspectionType.updateSuccess'))
      } else {
        await createInspectionTypeAPI(formData)
        ElMessage.success(t('mes.qc.inspectionType.createSuccess'))
      }
      router.push('/mes/inspectionType')
    } catch (error) {
      ElMessage.error(isEdit.value ? t('mes.qc.inspectionType.deleteFailed') : t('mes.qc.inspectionType.operationFailed'))
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
.inspection-type-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}
</style>
