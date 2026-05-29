<template>
  <div class="inspection-plan-detail">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.inspectionPlan.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.qc.inspectionPlan.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.id')">
          {{ formData.id }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.planCode')">
          {{ formData.planCode }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.planName')">
          {{ formData.planName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.inspectionType')">
          {{ formData.inspectionType }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.applicableProducts')" :span="2">
          {{ formData.applicableProducts }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.status')">
          <el-tag :type="getStatusType(formData.status || '')">
            {{ getStatusText(formData.status || '') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.version')">
          {{ formData.version }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.effectiveDate')">
          {{ formData.effectiveDate }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.expiryDate')">
          {{ formData.expiryDate }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.description')" :span="2">
          {{ formData.description }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.createdAt')">
          {{ formData.createdAt }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionPlan.updatedAt')">
          {{ formData.updatedAt }}
        </el-descriptions-item>
      </el-descriptions>

      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit">{{ t('common.edit') }}</el-button>
        <el-button @click="handleBack">{{ t('mes.qc.inspectionPlan.back') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { QcInspectionPlan } from '@/apis/qc'
import { getInspectionPlanByIdAPI } from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const loading = ref(false)
const formData = reactive<QcInspectionPlan>({
  id: 0,
  planCode: '',
  planName: '',
  inspectionType: '',
  applicableProducts: '',
  status: 'DRAFT',
  effectiveDate: '',
  expiryDate: '',
  version: 1,
  description: ''
})

const { t } = useI18n()

const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    SUSPENDED: 'warning',
    ARCHIVED: 'info'
  }
  return typeMap[status] || 'info'
}

const getStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    DRAFT: t('mes.qc.inspectionPlan.statusDraft'),
    ACTIVE: t('mes.qc.inspectionPlan.statusActive'),
    SUSPENDED: t('mes.qc.inspectionPlan.statusSuspended'),
    ARCHIVED: t('mes.qc.inspectionPlan.statusArchived')
  }
  return textMap[status] || status
}

const loadData = async () => {
  if (!route.query.id) return
  
  loading.value = true
  try {
    const res = await getInspectionPlanByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionPlan.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleEdit = () => {
  router.push({ path: '/mes/inspectionPlan/form', query: { id: route.query.id } })
}

const handleBack = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.inspection-plan-detail {
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