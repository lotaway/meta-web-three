<template>
  <div class="defect-code-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.defectCode.defectCode')">
          <el-input v-model="queryParams.defectCode" :placeholder="t('mes.qc.defectCode.defectCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.defectCode.defectName')">
          <el-input v-model="queryParams.defectName" :placeholder="t('mes.qc.defectCode.defectNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.defectCode.category')">
          <el-select v-model="queryParams.category" :placeholder="t('mes.qc.defectCode.categoryPlaceholder')" clearable>
            <el-option :label="t('mes.qc.defectCode.categoryAppearance')" value="外观" />
            <el-option :label="t('mes.qc.defectCode.categoryDimension')" value="尺寸" />
            <el-option :label="t('mes.qc.defectCode.categoryFunction')" value="功能" />
            <el-option :label="t('mes.qc.defectCode.categoryPerformance')" value="性能" />
            <el-option :label="t('mes.qc.defectCode.categoryMaterial')" value="材料" />
            <el-option :label="t('mes.qc.defectCode.categoryAssembly')" value="装配" />
            <el-option :label="t('mes.qc.defectCode.categoryOther')" value="其他" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.qc.defectCode.severity')">
          <el-select v-model="queryParams.severity" :placeholder="t('mes.qc.defectCode.severityPlaceholder')" clearable>
            <el-option :label="t('mes.qc.defectCode.severityCritical')" value="CRITICAL" />
            <el-option :label="t('mes.qc.defectCode.severityMajor')" value="MAJOR" />
            <el-option :label="t('mes.qc.defectCode.severityMinor')" value="MINOR" />
            <el-option :label="t('mes.qc.defectCode.severityObservation')" value="OBSERVATION" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
          <el-button @click="resetQuery">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.defectCode.add') }}</el-button>
      </div>

      <el-table :data="defectCodeList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.defectCode.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.defectCode.defectCode')" prop="defectCode" width="150" />
        <el-table-column :label="t('mes.qc.defectCode.defectName')" prop="defectName" width="180" />
        <el-table-column :label="t('mes.qc.defectCode.category')" prop="category" width="120">
          <template #default="{ row }">
            {{ getCategoryText(row.category) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.defectCode.severity')" prop="severity" width="120">
          <template #default="{ row }">
            <el-tag :type="getSeverityType(row.severity)">
              {{ getSeverityText(row.severity) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.defectCode.sortOrder')" prop="sortOrder" width="80" />
        <el-table-column :label="t('mes.qc.defectCode.status')" prop="isEnabled" width="100">
          <template #default="{ row }">
            <el-tag :type="row.isEnabled ? 'success' : 'info'">
              {{ row.isEnabled ? t('mes.qc.defectCode.statusActive') : t('mes.qc.defectCode.statusInactive') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.defectCode.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operations')" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.detail') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleEnable(row)" v-if="!row.isEnabled">{{ t('mes.qc.defectCode.enable') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDisable(row)" v-if="row.isEnabled">{{ t('mes.qc.defectCode.disable') }}</el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">{{ t('common.delete') }}</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="queryParams.pageNum"
        v-model:page-size="queryParams.pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { DefectCode, DefectCategory, DefectSeverity } from '@/apis/qc'
import {
  getDefectCodeListAPI,
  deleteDefectCodeAPI,
  enableDefectCodeAPI,
  disableDefectCodeAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const defectCodeList = ref<DefectCode[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  defectCode: '',
  defectName: '',
  category: '' as DefectCategory | '',
  severity: '' as DefectSeverity | ''
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

const getSeverityType = (severity: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const severityTypeMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    'CRITICAL': 'danger',
    'MAJOR': 'warning',
    'MINOR': 'info',
    'OBSERVATION': 'info'
  }
  return severityTypeMap[severity] || 'info'
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getDefectCodeListAPI()
    defectCodeList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.defectCode.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.defectCode = ''
  queryParams.defectName = ''
  queryParams.category = ''
  queryParams.severity = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/defectCode/form' })
}

const handleView = (row: DefectCode) => {
  router.push({ path: '/mes/defectCode/detail', query: { id: row.id } })
}

const handleEdit = (row: DefectCode) => {
  router.push({ path: '/mes/defectCode/form', query: { id: row.id } })
}

const handleEnable = async (row: DefectCode) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.defectCode.confirmEnable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await enableDefectCodeAPI(row.id!)
    ElMessage.success(t('mes.qc.defectCode.enableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.defectCode.enableFailed'))
    }
  }
}

const handleDisable = async (row: DefectCode) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.defectCode.confirmDisable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await disableDefectCodeAPI(row.id!)
    ElMessage.success(t('mes.qc.defectCode.disableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.defectCode.disableFailed'))
    }
  }
}

const handleDelete = async (row: DefectCode) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.defectCode.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteDefectCodeAPI(row.id!)
    ElMessage.success(t('mes.qc.defectCode.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.defectCode.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.defect-code-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>
