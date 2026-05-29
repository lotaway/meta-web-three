<template>
  <div class="inspection-type-detail">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.inspectionType.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.qc.inspectionType.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="t('common.id')">{{ detailData.id }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.typeCode')">{{ detailData.typeCode }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.typeName')">{{ detailData.typeName }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.category')">{{ getCategoryText(detailData.category) }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.status')" :span="2">
          <el-tag :type="detailData.status === 'ACTIVE' ? 'success' : 'info'">
            {{ detailData.status === 'ACTIVE' ? t('mes.qc.inspectionType.statusActive') : t('mes.qc.inspectionType.statusInactive') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.equipment.description')" :span="2">{{ detailData.description || '-' }}</el-descriptions-item>
        <el-descriptions-item label="Applicable Products" :span="2">{{ detailData.applicableProducts || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.defaultSamplingPlan')">{{ detailData.defaultSamplingPlan || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.defaultAql')">{{ detailData.defaultAql || '-' }}</el-descriptions-item>
        <el-descriptions-item label="Default Inspection Timeout (Hours)">
          {{ detailData.defaultTimeoutHours ? detailData.defaultTimeoutHours + ' hours' : '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.sortOrder')">{{ detailData.sortOrder || 0 }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.requireCertificate')">
          {{ detailData.requireCertificate ? t('mes.qc.inspectionType.yes') : t('mes.qc.inspectionType.no') }}
        </el-descriptions-item>
        <el-descriptions-item label="Require Test Report">
          {{ detailData.requireTestReport ? t('mes.qc.inspectionType.yes') : t('mes.qc.inspectionType.no') }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.createdAt')">{{ detailData.createdAt || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionType.updatedAt')">{{ detailData.updatedAt || '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import type { QcInspectionType } from '@/apis/qc'
import { getInspectionTypeByIdAPI } from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const detailData = reactive<QcInspectionType>({} as QcInspectionType)

const getCategoryText = (category: string) => {
  const categoryMap: Record<string, string> = {
    INCOMING: t('mes.qc.inspectionType.categoryIncoming'),
    PROCESS: t('mes.qc.inspectionType.categoryProcess'),
    FINAL: t('mes.qc.inspectionType.categoryFinal'),
    OUTGOING: t('mes.qc.inspectionType.categoryOutgoing'),
    CUSTOM: t('mes.qc.inspectionType.categoryCustom')
  }
  return categoryMap[category] || category
}

const loadData = async () => {
  if (!route.query.id) {
    ElMessage.error(t('mes.processRoute.paramError'))
    router.push('/mes/inspectionType')
    return
  }
  
  loading.value = true
  try {
    const res = await getInspectionTypeByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(detailData, res.data)
    } else {
      ElMessage.error(t('mes.processRoute.dataNotExist'))
      router.push('/mes/inspectionType')
    }
  } catch (error) {
    ElMessage.error(t('mes.processRoute.loadFailed'))
    router.push('/mes/inspectionType')
  } finally {
    loading.value = false
  }
}

const handleBack = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.inspection-type-detail {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
}
</style>
