<template>
  <div class="inspection-item-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.qc.inspectionItem.itemCode')">
          <el-input v-model="queryParams.itemCode" :placeholder="t('mes.qc.inspectionItem.itemCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionItem.itemName')">
          <el-input v-model="queryParams.itemName" :placeholder="t('mes.qc.inspectionItem.itemNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.qc.inspectionItem.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.qc.inspectionItem.statusPlaceholder')" clearable>
            <el-option :label="t('mes.qc.inspectionItem.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.inspectionItem.statusInactive')" value="INACTIVE" />
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
        <el-button type="primary" @click="handleAdd">{{ t('mes.qc.inspectionItem.add') }}</el-button>
      </div>

      <el-table :data="itemList" v-loading="loading" border stripe>
        <el-table-column :label="t('mes.qc.inspectionItem.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.qc.inspectionItem.itemCode')" prop="itemCode" width="150" />
        <el-table-column :label="t('mes.qc.inspectionItem.itemName')" prop="itemName" width="180" />
        <el-table-column :label="t('mes.qc.inspectionItem.inspectionMethod')" prop="inspectionMethod" width="150" />
        <el-table-column :label="t('mes.qc.inspectionItem.equipmentRequired')" prop="equipmentRequired" width="150" />
        <el-table-column :label="t('mes.qc.inspectionItem.standardValue')" prop="standardValue" width="120" />
        <el-table-column :label="t('mes.qc.inspectionItem.unit')" prop="unit" width="80" />
        <el-table-column :label="t('mes.qc.inspectionItem.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'">
              {{ row.status === 'ACTIVE' ? t('mes.qc.inspectionItem.statusActive') : t('mes.qc.inspectionItem.statusInactive') }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.qc.inspectionItem.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operation')" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="success" size="small" @click="handleActivate(row)" v-if="row.status === 'INACTIVE'">{{ t('mes.qc.inspectionItem.activate') }}</el-button>
            <el-button link type="warning" size="small" @click="handleDeactivate(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.qc.inspectionItem.deactivate') }}</el-button>
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
import type { QcInspectionItem } from '@/apis/qc'
import {
  getInspectionItemListAPI,
  deleteInspectionItemAPI,
  activateInspectionItemAPI,
  deactivateInspectionItemAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const itemList = ref<QcInspectionItem[]>([])
const total = ref(0)

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  itemCode: '',
  itemName: '',
  status: ''
})

const getList = async () => {
  loading.value = true
  try {
    const res = await getInspectionItemListAPI()
    itemList.value = res.data || []
    total.value = res.data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.qc.inspectionItem.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.itemCode = ''
  queryParams.itemName = ''
  queryParams.status = ''
  getList()
}

const handleAdd = () => {
  router.push({ path: '/mes/inspectionItem/form' })
}

const handleView = (row: QcInspectionItem) => {
  router.push({ path: '/mes/inspectionItem/detail', query: { id: row.id } })
}

const handleEdit = (row: QcInspectionItem) => {
  router.push({ path: '/mes/inspectionItem/form', query: { id: row.id } })
}

const handleActivate = async (row: QcInspectionItem) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionItem.confirmActivate'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await activateInspectionItemAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionItem.activateSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionItem.activateFailed'))
    }
  }
}

const handleDeactivate = async (row: QcInspectionItem) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionItem.confirmDeactivate'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deactivateInspectionItemAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionItem.deactivateSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionItem.deactivateFailed'))
    }
  }
}

const handleDelete = async (row: QcInspectionItem) => {
  try {
    await ElMessageBox.confirm(t('mes.qc.inspectionItem.confirmDelete'), t('message.prompt'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await deleteInspectionItemAPI(row.id!)
    ElMessage.success(t('mes.qc.inspectionItem.deleteSuccess'))
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.qc.inspectionItem.deleteFailed'))
    }
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.inspection-item-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.toolbar {
  margin-bottom: 20px;
}
</style>