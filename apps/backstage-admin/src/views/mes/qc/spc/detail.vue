<template>
  <div class="spc-control-chart-detail">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.spc.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.qc.spc.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="t('mes.qc.spc.id')">
          {{ formData.id }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.chartCode')">
          {{ formData.chartCode }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.chartName')">
          {{ formData.chartName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.chartType')">
          {{ getChartTypeText(formData.chartType || '') }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.parameterName')">
          {{ formData.parameterName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.unit')">
          {{ formData.unit }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.targetValue')">
          {{ formData.targetValue }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.usl')">
          {{ formData.usl }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.lsl')">
          {{ formData.lsl }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.ucl')">
          {{ formData.ucl }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.lcl')">
          {{ formData.lcl }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.centerLine')">
          {{ formData.centerLine }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.status')">
          <el-tag :type="formData.status === 'ACTIVE' ? 'success' : 'info'">
            {{ formData.status === 'ACTIVE' ? t('mes.qc.spc.statusActive') : t('mes.qc.spc.statusInactive') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.description')" :span="2">
          {{ formData.description }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.createdAt')">
          {{ formData.createdAt }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.spc.updatedAt')">
          {{ formData.updatedAt }}
        </el-descriptions-item>
      </el-descriptions>

      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit">{{ t('common.edit') }}</el-button>
        <el-button @click="handleBack">{{ t('mes.qc.spc.back') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { SpcControlChart } from '@/apis/qc'
import { getSpcControlChartByIdAPI } from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const loading = ref(false)
const formData = reactive<SpcControlChart>({
  id: 0,
  chartCode: '',
  chartName: '',
  chartType: 'XBAR_R',
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

const getChartTypeText = (type: string) => {
  const textMap: Record<string, string> = {
    XBAR_R: t('mes.qc.spc.chartTypeXbarR'),
    XBAR_S: t('mes.qc.spc.chartTypeXbarS'),
    X_MR: t('mes.qc.spc.chartTypeXMr'),
    P_CHART: t('mes.qc.spc.chartTypePChart'),
    NP_CHART: t('mes.qc.spc.chartTypeNpChart'),
    C_CHART: t('mes.qc.spc.chartTypeCChart'),
    U_CHART: t('mes.qc.spc.chartTypeUChart')
  }
  return textMap[type] || type
}

const loadData = async () => {
  if (!route.query.id) return
  
  loading.value = true
  try {
    const res = await getSpcControlChartByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.spc.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleEdit = () => {
  router.push({ path: '/mes/spc/form', query: { id: route.query.id } })
}

const handleBack = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.spc-control-chart-detail {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
}

.action-buttons {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}
</style>