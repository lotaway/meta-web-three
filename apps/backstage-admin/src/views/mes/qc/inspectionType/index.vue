<template>
  <div class="inspection-type-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.inspectionType.typeCode')">
          <el-input v-model="queryParams.typeCode" :placeholder="t('mes.qc.inspectionType.typeCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionType.typeName')">
          <el-input v-model="queryParams.typeName" :placeholder="t('mes.qc.inspectionType.typeNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionType.category')">
          <el-select v-model="queryParams.category" :placeholder="t('mes.qc.inspectionType.categoryPlaceholder')" clearable>
            <el-option :label="t('mes.qc.inspectionType.categoryIncoming')" value="INCOMING" />
            <el-option :label="t('mes.qc.inspectionType.categoryProcess')" value="PROCESS" />
            <el-option :label="t('mes.qc.inspectionType.categoryFinal')" value="FINAL" />
            <el-option :label="t('mes.qc.inspectionType.categoryOutgoing')" value="OUTGOING" />
            <el-option :label="t('mes.qc.inspectionType.categoryCustom')" value="CUSTOM" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionType.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.qc.inspectionType.statusPlaceholder')" clearable>
            <el-option :label="t('mes.qc.inspectionType.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.inspectionType.statusInactive')" value="INACTIVE" />
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
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.inspectionType.add') }}</el-button>
      </div>

      <el-table :data="typeList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.inspectionType.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.inspectionType.typeCode')" prop="typeCode" width="150" />
        <el-table-column :label="t('mes.qc.inspectionType.typeName')" prop="typeName" width="180" />
        <el-table-column :label="t('mes.qc.inspectionType.category')" prop="category" width="120">
          <template #default="{ row }">
            {{ getCategoryText(row.category) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.inspectionType.defaultSamplingPlan')" prop="defaultSamplingPlan" width="150" />
        <el-table-column :label="t('mes.qc.inspectionType.defaultAql')" prop="defaultAql" width="100" />
        <el-table-column :label="t('mes.qc.inspectionType.requireCertificate')" prop="requireCertificate" width="100">
          <template #default="{ row }">
            {{ row.requireCertificate ? t('mes.qc.inspectionType.yes') : t('mes.qc.inspectionType.no') }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.inspectionType.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'">
              {{ row.status === 'ACTIVE' ? t('mes.qc.inspectionType.statusActive') : t('mes.qc.inspectionType.statusInactive') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.inspectionType.sortOrder')" prop="sortOrder" width="80" />
        <el-table-column :label="t('mes.qc.inspectionType.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operation')" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleActivate(row)" v-if="row.status === 'INACTIVE'">{{ t('mes.qc.inspectionType.activate') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDeactivate(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.qc.inspectionType.deactivate') }}</el-button>
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
import type { QcInspectionType, InspectionCategory } from '@/apis/qc'
import {
  getInspectionTypeListAPI,
  deleteInspectionTypeAPI,
  activateInspectionTypeAPI,
  deactivateInspectionTypeAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const typeList = ref<QcInspectionType[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  typeCode: '',
  typeName: '',
  category: '' as InspectionCategory | '',
  status: ''
})

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

const getList = async () => {
  loading.value = true
  try {
    const res = await getInspectionTypeListAPI()
    typeList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionType.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.typeCode = ''
  queryParams.typeName = ''
  queryParams.category = ''
  queryParams.status = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/inspectionType/form' })
}

const handleView = (row: QcInspectionType) => {
  router.push({ path: '/mes/inspectionType/detail', query: { id: row.id } })
}

const handleEdit = (row: QcInspectionType) => {
  router.push({ path: '/mes/inspectionType/form', query: { id: row.id } })
}

const handleActivate = async (row: QcInspectionType) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionType.confirmActivate'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await activateInspectionTypeAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionType.activateSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionType.activateFailed'))
    }
  }
}

const handleDeactivate = async (row: QcInspectionType) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionType.confirmDeactivate'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deactivateInspectionTypeAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionType.deactivateSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionType.deactivateFailed'))
    }
  }
}

const handleDelete = async (row: QcInspectionType) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionType.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteInspectionTypeAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionType.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionType.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.inspection-type-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>
