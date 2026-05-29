<template>
  <div class="inspection-item-detail">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.qc.inspectionItem.detail') }}</span>
          <el-button @click="handleBack">{{ t('mes.qc.inspectionItem.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="t('mes.qc.inspectionItem.id')">
          {{ formData.id }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.itemCode')">
          {{ formData.itemCode }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.itemName')">
          {{ formData.itemName }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.inspectionMethod')">
          {{ formData.inspectionMethod }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.equipmentRequired')">
          {{ formData.equipmentRequired }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.standardValue')">
          {{ formData.standardValue }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.upperLimit')">
          {{ formData.upperLimit }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.lowerLimit')">
          {{ formData.lowerLimit }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.unit')">
          {{ formData.unit }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.status')">
          <el-tag :type="formData.status === 'ACTIVE' ? 'success' : 'info'">
            {{ formData.status === 'ACTIVE' ? t('mes.qc.inspectionItem.statusActive') : t('mes.qc.inspectionItem.statusInactive') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.createdAt')">
          {{ formData.createdAt }}
        </el-descriptions-item>
        <el-descriptions-item :label="t('mes.qc.inspectionItem.updatedAt')">
          {{ formData.updatedAt }}
        </el-descriptions-item>
      </el-descriptions>

      <div class="action-buttons">
        <el-button type="primary" @click="handleEdit">{{ t('common.edit') }}</el-button>
        <el-button @click="handleBack">{{ t('mes.qc.inspectionItem.back') }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { QcInspectionItem } from '@/apis/qc'
import { getInspectionItemByIdAPI } from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const loading = ref(false)
const formData = reactive<QcInspectionItem>({
  id: 0,
  itemCode: '',
  itemName: '',
  inspectionMethod: '',
  equipmentRequired: '',
  standardValue: '',
  upperLimit: undefined,
  lowerLimit: undefined,
  unit: '',
  status: 'INACTIVE'
})

const { t } = useI18n()

const loadData = async () => {
  if (!route.query.id) return
  
  loading.value = true
  try {
    const res = await getInspectionItemByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionItem.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleEdit = () => {
  router.push({ path: '/mes/inspectionItem/form', query: { id: route.query.id } })
}

const handleBack = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.inspection-item-detail {
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