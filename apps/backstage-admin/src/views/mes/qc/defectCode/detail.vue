<template>
  <div class="defect-code-detail-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.defectCode.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.qc.defectCode.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="t('mes.qc.defectCode.id')">
          {{ detailData.id }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.defectCode')">
          {{ detailData.defectCode }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.defectName')">
          {{ detailData.defectName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.category')">
          {{ getCategoryText(detailData.category) }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.severity')">
          <el-tag :type="getSeverityType(detailData.severity)">
            {{ getSeverityText(detailData.severity) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.status')">
          <el-tag :type="detailData.isEnabled ? 'success' : 'info'">
            {{ detailData.isEnabled ? t('mes.qc.defectCode.statusActive') : t('mes.qc.defectCode.statusInactive') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.description')" :span="2">
          {{ detailData.description || '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.dispositionGuide')" :span="2">
          {{ detailData.dispositionGuide || '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.sortOrder')">
          {{ detailData.sortOrder }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.createdAt')">
          {{ detailData.createdAt }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.defectCode.updatedAt')">
          {{ detailData.updatedAt }}
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import type { DefectCode } from '@/apis/qc'
import { getDefectCodeByIdAPI } from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const detailData = reactive<DefectCode>({
  id: undefined,
  defectCode: '',
  defectName: '',
  category: '其他',
  severity: 'MINOR',
  description: '',
  dispositionGuide: '',
  sortOrder: 0,
  isEnabled: true,
  createdAt: '',
  updatedAt: ''
})

const getCategoryText = (category: string) => {
  const categoryMap: Record<string, string> = {
    '外观': t('mes.qc.defectCode.categoryAppearance'),
    '尺寸': t('mes.qc.defectCode.categoryDimension'),
    '功能': t('mes.qc.defectCode.categoryFunction'),
    '性能': t('mes.qc.defectCode.categoryPerformance'),
    '材料': t('mes.qc.defectCode.categoryMaterial'),
    '装配': t('mes.qc.defectCode.categoryAssembly'),
    '其他': t('mes.qc.defectCode.categoryOther')
  }
  return categoryMap[category] || category
}

const getSeverityText = (severity: string) => {
  const severityMap: Record<string, string> = {
    'CRITICAL': t('mes.qc.defectCode.severityCritical'),
    'MAJOR': t('mes.qc.defectCode.severityMajor'),
    'MINOR': t('mes.qc.defectCode.severityMinor'),
    'OBSERVATION': t('mes.qc.defectCode.severityObservation')
  }
  return severityMap[severity] || severity
}

const getSeverityType = (severity: string) => {
  const severityTypeMap: Record<string, string> = {
    'CRITICAL': 'danger',
    'MAJOR': 'warning',
    'MINOR': 'info',
    'OBSERVATION': ''
  }
  return severityTypeMap[severity] || 'info'
}

const loadData = async () => {
  if (!route.query.id) {
    ElMessage.error(t('message.operationFailed'))
    return
  }
  loading.value = true
  try {
    const res = await getDefectCodeByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(detailData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('message.operationFailed'))
  } finally {
    loading.value = false
  }
}

const handleBack = () => {
  router.push({ path: '/mes/defectCode' })
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.defect-code-detail-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: bold;
}
</style>
