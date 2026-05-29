<template>
  <div class="defect-code-form-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.defectCode.edit') : t('mes.qc.defectCode.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="150px">
        <el-form-item :label="t('mes.qc.defectCode.defectCode')" prop="defectCode">
          <el-input v-model="formData.defectCode" :placeholder="t('mes.qc.defectCode.defectCodePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.defectCode.defectName')" prop="defectName">
          <el-input v-model="formData.defectName" :placeholder="t('mes.qc.defectCode.defectNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.defectCode.category')" prop="category">
          <el-select v-model="formData.category" :placeholder="t('mes.qc.defectCode.categoryPlaceholder')">
            <el-option :label="t('mes.qc.defectCode.categoryAppearance')" value="外观" />
            <el-option :label="t('mes.qc.defectCode.categoryDimension')" value="尺寸" />
            <el-option :label="t('mes.qc.defectCode.categoryFunction')" value="功能" />
            <el-option :label="t('mes.qc.defectCode.categoryPerformance')" value="性能" />
            <el-option :label="t('mes.qc.defectCode.categoryMaterial')" value="材料" />
            <el-option :label="t('mes.qc.defectCode.categoryAssembly')" value="装配" />
            <el-option :label="t('mes.qc.defectCode.categoryOther')" value="其他" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.defectCode.severity')" prop="severity">
          <el-select v-model="formData.severity" :placeholder="t('mes.qc.defectCode.severityPlaceholder')">
            <el-option :label="t('mes.qc.defectCode.severityCritical')" value="CRITICAL" />
            <el-option :label="t('mes.qc.defectCode.severityMajor')" value="MAJOR" />
            <el-option :label="t('mes.qc.defectCode.severityMinor')" value="MINOR" />
            <el-option :label="t('mes.qc.defectCode.severityObservation')" value="OBSERVATION" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.defectCode.description')" prop="description">
          <el-input v-model="formData.description" type="textarea" :rows="3" :placeholder="t('mes.qc.defectCode.descriptionPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.defectCode.dispositionGuide')" prop="dispositionGuide">
          <el-input v-model="formData.dispositionGuide" type="textarea" :rows="3" :placeholder="t('mes.qc.defectCode.dispositionGuidePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.defectCode.sortOrder')" prop="sortOrder">
          <el-input-number v-model="formData.sortOrder" :min="0" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit">{{ t('common.save') }}</el-button>
          <el-button @click="handleBack">{{ t('common.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, FormInstance, FormRules } from 'element-plus'
import type { DefectCode, DefectCategory, DefectSeverity } from '@/apis/qc'
import {
  getDefectCodeByIdAPI,
  createDefectCodeAPI,
  updateDefectCodeAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const loading = ref(false)

const isEdit = computed(() => !!route.query.id)

const formData = reactive<DefectCode>({
  defectCode: '',
  defectName: '',
  category: '其他' as DefectCategory,
  severity: 'MINOR' as DefectSeverity,
  description: '',
  dispositionGuide: '',
  sortOrder: 0,
  isEnabled: true
})

const rules = reactive<FormRules>({
  defectCode: [
    { required: true, message: t('mes.qc.defectCode.defectCodePlaceholder'), trigger: 'blur' }
  ],
  defectName: [
    { required: true, message: t('mes.qc.defectCode.defectNamePlaceholder'), trigger: 'blur' }
  ],
  category: [
    { required: true, message: t('mes.qc.defectCode.categoryPlaceholder'), trigger: 'change' }
  ],
  severity: [
    { required: true, message: t('mes.qc.defectCode.severityPlaceholder'), trigger: 'change' }
  ]
})

const loadData = async () => {
  if (!route.query.id) return
  loading.value = true
  try {
    const res = await getDefectCodeByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('message.operationFailed'))
  } finally {
    loading.value = false
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      if (isEdit.value) {
        await updateDefectCodeAPI(Number(route.query.id), formData)
      } else {
        await createDefectCodeAPI(formData)
      }
      ElMessage.success(t('message.saveSuccess'))
      handleBack()
    } catch (error) {
      ElMessage.error(t('message.saveFailed'))
    } finally {
      loading.value = false
    }
  })
}

const handleBack = () => {
  router.push({ path: '/mes/defectCode' })
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.defect-code-form-container {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: bold;
}
</style>
