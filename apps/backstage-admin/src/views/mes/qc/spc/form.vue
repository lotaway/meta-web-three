<template>
  <div class="spc-control-chart-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.spc.edit') : t('mes.qc.spc.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="180px">
        <el-form-item :label="t('mes.qc.spc.chartCode')" prop="chartCode">
          <el-input v-model="formData.chartCode" :disabled="isEdit" :placeholder="t('mes.qc.spc.chartCodePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.chartName')" prop="chartName">
          <el-input v-model="formData.chartName" :placeholder="t('mes.qc.spc.chartNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.chartType')" prop="chartType">
          <el-select v-model="formData.chartType" :placeholder="t('mes.qc.spc.chartTypePlaceholder')">
            <el-option :label="t('mes.qc.spc.chartTypeXbarR')" value="XBAR_R" />
            <el-option :label="t('mes.qc.spc.chartTypeXbarS')" value="XBAR_S" />
            <el-option :label="t('mes.qc.spc.chartTypeXMr')" value="X_MR" />
            <el-option :label="t('mes.qc.spc.chartTypePChart')" value="P_CHART" />
            <el-option :label="t('mes.qc.spc.chartTypeNpChart')" value="NP_CHART" />
            <el-option :label="t('mes.qc.spc.chartTypeCChart')" value="C_CHART" />
            <el-option :label="t('mes.qc.spc.chartTypeUChart')" value="U_CHART" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.parameterName')" prop="parameterName">
          <el-input v-model="formData.parameterName" :placeholder="t('mes.qc.spc.parameterNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.unit')" prop="unit">
          <el-input v-model="formData.unit" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.targetValue')" prop="targetValue">
          <el-input-number v-model="formData.targetValue" :precision="3" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.usl')" prop="usl">
          <el-input-number v-model="formData.usl" :precision="3" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.lsl')" prop="lsl">
          <el-input-number v-model="formData.lsl" :precision="3" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.ucl')" prop="ucl">
          <el-input-number v-model="formData.ucl" :precision="3" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.lcl')" prop="lcl">
          <el-input-number v-model="formData.lcl" :precision="3" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.centerLine')" prop="centerLine">
          <el-input-number v-model="formData.centerLine" :precision="3" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.status')" prop="status">
          <el-select v-model="formData.status" :placeholder="t('mes.qc.spc.statusPlaceholder')">
            <el-option :label="t('mes.qc.spc.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.spc.statusInactive')" value="INACTIVE" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.spc.description')" prop="description">
          <el-input v-model="formData.description" type="textarea" :rows="3" :placeholder="t('mes.qc.spc.descriptionPlaceholder')" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('mes.qc.spc.save') }}</el-button>
          <el-button @click="handleCancel">{{ t('mes.qc.spc.cancel') }}</el-button>
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
import type { SpcControlChart, ChartType } from '@/apis/qc'
import {
  getSpcControlChartByIdAPI,
  createSpcControlChartAPI,
  updateSpcControlChartAPI
} from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const formData = reactive<SpcControlChart>({
  chartCode: '',
  chartName: '',
  chartType: 'XBAR_R' as ChartType,
  parameterName: '',
  unit: '',
  targetValue: undefined,
  usl: undefined,
  lsl: undefined,
  ucl: undefined,
  lcl: undefined,
  centerLine: undefined,
  status: 'INACTIVE',
  description: ''
})

const { t } = useI18n()

const rules = {
  chartCode: [
    { required: true, message: t('mes.qc.spc.chartCodePlaceholder'), trigger: 'blur' }
  ],
  chartName: [
    { required: true, message: t('mes.qc.spc.chartNamePlaceholder'), trigger: 'blur' }
  ],
  chartType: [
    { required: true, message: t('mes.qc.spc.chartTypePlaceholder'), trigger: 'change' }
  ],
  parameterName: [
    { required: true, message: t('mes.qc.spc.parameterNamePlaceholder'), trigger: 'blur' }
  ]
}

const loadData = async () => {
  if (!route.query.id) return
  
  try {
    const res = await getSpcControlChartByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.spc.loadFailed'))
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    submitting.value = true
    try {
      if (isEdit.value) {
        await updateSpcControlChartAPI(Number(route.query.id), formData)
        ElMessage.success(t('mes.qc.spc.updateSuccess'))
      } else {
        await createSpcControlChartAPI(formData)
        ElMessage.success(t('mes.qc.spc.createSuccess'))
      }
      router.push('/mes/spc')
    } catch (error) {
      ElMessage.error(t('mes.qc.spc.deleteFailed'))
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
.spc-control-chart-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}
</style>