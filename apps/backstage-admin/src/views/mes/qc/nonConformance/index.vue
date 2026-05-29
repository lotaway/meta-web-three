<template>
  <div class="non-conformance-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.nonConformance.dispositionCode')">
          <el-input v-model="queryParams.dispositionCode" :placeholder="t('mes.qc.nonConformance.dispositionCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.nonConformance.dispositionName')">
          <el-input v-model="queryParams.dispositionName" :placeholder="t('mes.qc.nonConformance.dispositionNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.nonConformance.type')">
          <el-select v-model="queryParams.type" :placeholder="t('mes.qc.nonConformance.typePlaceholder')" clearable>
            <el-option :label="t('mes.qc.nonConformance.typeScrap')" value="SCRAP" />
            <el-option :label="t('mes.qc.nonConformance.typeRework')" value="REWORK" />
            <el-option :label="t('mes.qc.nonConformance.typeReturn')" value="RETURN" />
            <el-option :label="t('mes.qc.nonConformance.typeUseAsIs')" value="USE_AS_IS" />
            <el-option :label="t('mes.qc.nonConformance.type降级使用')" value="降级使用" />
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
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.nonConformance.add') }}</el-button>
      </div>

      <el-table :data="dispositionList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.nonConformance.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.nonConformance.dispositionCode')" prop="dispositionCode" width="150" />
        <el-table-column :label="t('mes.qc.nonConformance.dispositionName')" prop="dispositionName" width="200" />
        <el-table-column :label="t('mes.qc.nonConformance.type')" prop="type" width="120">
          <template #default="{ row }">
            {{ getTypeText(row.type) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.nonConformance.steps')" prop="steps" width="100">
          <template #default="{ row }">
            {{ row.steps?.length || 0 }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.nonConformance.isEnabled')" prop="isEnabled" width="100">
          <template #default="{ row }">
            <el-tag :type="row.isEnabled ? 'success' : 'info'">
              {{ row.isEnabled ? t('common.enable') : t('common.disable') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.nonConformance.sortOrder')" prop="sortOrder" width="80" />
        <el-table-column :label="t('mes.qc.nonConformance.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operation')" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleEnable(row)" v-if="!row.isEnabled">{{ t('mes.qc.nonConformance.enable') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDisable(row)" v-if="row.isEnabled">{{ t('mes.qc.nonConformance.disable') }}</el-button>
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
import type { NonConformanceDisposition, DispositionType } from '@/apis/qc'
import {
  getNonConformanceDispositionListAPI,
  deleteNonConformanceDispositionAPI,
  enableNonConformanceDispositionAPI,
  disableNonConformanceDispositionAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const dispositionList = ref<NonConformanceDisposition[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  dispositionCode: '',
  dispositionName: '',
  type: '' as DispositionType | ''
})

const getTypeText = (type: string) => {
  const typeMap: Record<string, string> = {
    SCRAP: t('mes.qc.nonConformance.typeScrap'),
    REWORK: t('mes.qc.nonConformance.typeRework'),
    RETURN: t('mes.qc.nonConformance.typeReturn'),
    USE_AS_IS: t('mes.qc.nonConformance.typeUseAsIs'),
    '降级使用': t('mes.qc.nonConformance.type降级使用')
  }
  return typeMap[type] || type
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getNonConformanceDispositionListAPI()
    dispositionList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.nonConformance.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.dispositionCode = ''
  queryParams.dispositionName = ''
  queryParams.type = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/nonConformance/form' })
}

const handleView = (row: NonConformanceDisposition) => {
  router.push({ path: '/mes/nonConformance/detail', query: { id: row.id } })
}

const handleEdit = (row: NonConformanceDisposition) => {
  router.push({ path: '/mes/nonConformance/form', query: { id: row.id } })
}

const handleEnable = async (row: NonConformanceDisposition) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.nonConformance.confirmEnable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await enableNonConformanceDispositionAPI(row.id!)
    ElMessage.success(t('mes.qc.nonConformance.enableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.nonConformance.enableFailed'))
    }
  }
}

const handleDisable = async (row: NonConformanceDisposition) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.nonConformance.confirmDisable'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await disableNonConformanceDispositionAPI(row.id!)
    ElMessage.success(t('mes.qc.nonConformance.disableSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.nonConformance.disableFailed'))
    }
  }
}

const handleDelete = async (row: NonConformanceDisposition) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.nonConformance.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteNonConformanceDispositionAPI(row.id!)
    ElMessage.success(t('mes.qc.nonConformance.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.nonConformance.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.non-conformance-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>