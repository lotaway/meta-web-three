<template>
  <div class="inspection-type-detail">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ $t('mes.inspectionType.detail.title') }}</span>
          <el-button @click="handleBack">{{ $t('common.back') }}</el-button>
        </div>
      </template>

      <el-descriptions :column="2" border v-loading="loading">
        <el-descriptions-item :label="$t('common.id')">{{ detailData.id }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.typeCode')">{{ detailData.typeCode }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.typeName')">{{ detailData.typeName }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.category')">{{ getCategoryText(detailData.category) }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.status')" :span="2">
          <el-tag :type="detailData.status === 'ACTIVE' ? 'success' : 'info'">
            {{ detailData.status === 'ACTIVE' ? $t('mes.inspectionType.detail.active') : $t('mes.inspectionType.detail.inactive') }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.description')" :span="2">{{ detailData.description || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.applicableProducts')" :span="2">{{ detailData.applicableProducts || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.defaultSamplingPlan')">{{ detailData.defaultSamplingPlan || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.defaultAql')">{{ detailData.defaultAql || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.defaultTimeoutHours')">
          {{ detailData.defaultTimeoutHours ? detailData.defaultTimeoutHours + ' ' + $t('mes.inspectionType.detail.hours') : '-' }}
        </el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.sortOrder')">{{ detailData.sortOrder || 0 }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.requireCertificate')">
          {{ detailData.requireCertificate ? $t('common.yes') : $t('common.no') }}
        </el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.requireTestReport')">
          {{ detailData.requireTestReport ? $t('common.yes') : $t('common.no') }}
        </el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.createdAt')">{{ detailData.createdAt || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="$t('mes.inspectionType.detail.updatedAt')">{{ detailData.updatedAt || '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import type { QcInspectionType } from '@/apis/qc'
import { getInspectionTypeByIdAPI } from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const loading = ref(false)
const detailData = reactive<QcInspectionType>({} as QcInspectionType)

const getCategoryText = (category: string) => {
  const categoryMap: Record<string, string> = {
    INCOMING: '来料检验',
    PROCESS: '工序检验',
    FINAL: '最终检验',
    OUTGOING: '出货检验',
    CUSTOM: '自定义'
  }
  return categoryMap[category] || category
}

const loadData = async () => {
  if (!route.query.id) {
    ElMessage.error('缺少ID参数')
    router.push('/mes/inspectionType')
    return
  }
  
  loading.value = true
  try {
    const res = await getInspectionTypeByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(detailData, res.data)
    } else {
      ElMessage.error('数据不存在')
      router.push('/mes/inspectionType')
    }
  } catch (error) {
    ElMessage.error('加载数据失败')
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
