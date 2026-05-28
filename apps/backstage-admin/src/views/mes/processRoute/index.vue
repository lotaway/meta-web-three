<template>
  <div class="process-route-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item :label="t('mes.processRoute.routeCode')">
          <el-input v-model="queryParams.routeCode" :placeholder="t('mes.processRoute.routeCodePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.processRoute.routeName')">
          <el-input v-model="queryParams.routeName" :placeholder="t('mes.processRoute.routeNamePlaceholder')" clearable />
        </el-form-item>
        <el-form-item :label="t('mes.processRoute.status')">
          <el-select v-model="queryParams.status" :placeholder="t('mes.processRoute.statusPlaceholder')" clearable>
            <el-option :label="t('mes.processRoute.statusDraft')" value="DRAFT" />
            <el-option :label="t('mes.processRoute.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.processRoute.statusArchived')" value="ARCHIVED" />
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
        <el-button type="primary" @click="handleAdd">{{ t('mes.processRoute.add') }}</el-button>
      </div>

      <el-table :data="routeList" v-loading="loading" border stripe>
        <el-table-column :label="t('common.id')" prop="id" width="80" />
        <el-table-column :label="t('mes.processRoute.routeCode')" prop="routeCode" width="150" />
        <el-table-column :label="t('mes.processRoute.routeName')" prop="routeName" width="180" />
        <el-table-column :label="t('mes.processRoute.productCode')" prop="productCode" width="150" />
        <el-table-column :label="t('mes.processRoute.version')" prop="version" width="80" />
        <el-table-column :label="t('mes.processRoute.status')" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processRoute.stepCount')" width="100">
          <template #default="{ row }">
            {{ row.steps?.length || 0 }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processRoute.createdAt')" prop="createdAt" width="180" />
        <el-table-column :label="t('common.operation')" fixed="right" width="240">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-button link type="warning" size="small" @click="handleActivate(row)" v-if="row.status === 'DRAFT'">{{ t('mes.processRoute.activate') }}</el-button>
            <el-button link type="info" size="small" @click="handleArchive(row)" v-if="row.status === 'ACTIVE'">{{ t('mes.processRoute.archive') }}</el-button>
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
import type { ProcessRoute } from '@/apis/processRoute'
import { 
  getProcessRouteListAPI, 
  deleteProcessRouteAPI,
  activateProcessRouteAPI,
  archiveProcessRouteAPI 
} from '@/apis/processRoute'

const { t } = useI18n()

const router = useRouter()

const loading = ref(false)
const routeList = ref<ProcessRoute[]>([])
const total = ref(0)

const queryParams = reactive({
  routeCode: '',
  routeName: '',
  status: '',
  pageNum: 1,
  pageSize: 10
})

const getList = async () => {
  loading.value = true
  try {
    const res = await getProcessRouteListAPI(queryParams.status || undefined)
    const data = res.data
    routeList.value = data || []
    total.value = data?.length || 0
  } catch (error) {
    ElMessage.error(t('mes.processRoute.fetchListFailed'))
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const resetQuery = () => {
  queryParams.routeCode = ''
  queryParams.routeName = ''
  queryParams.status = ''
  handleQuery()
}

const handleAdd = () => {
  router.push('/mes/process-route/form')
}

const handleEdit = (row: ProcessRoute) => {
  router.push({ path: '/mes/process-route/form', query: { id: row.id } })
}

const handleView = (row: ProcessRoute) => {
  router.push({ path: '/mes/process-route/detail', query: { id: row.id } })
}

const handleActivate = async (row: ProcessRoute) => {
  try {
    await ElMessageBox.confirm(t('mes.processRoute.confirmActivate'), t('common.warning'), {
      type: 'warning'
    })
    await activateProcessRouteAPI(row.id!)
    ElMessage.success(t('mes.processRoute.activateSuccess'))
    getList()
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.processRoute.activateFailed'))
    }
  }
}

const handleArchive = async (row: ProcessRoute) => {
  try {
    await ElMessageBox.confirm(t('mes.processRoute.confirmArchive'), t('common.warning'), {
      type: 'warning'
    })
    await archiveProcessRouteAPI(row.id!)
    ElMessage.success(t('mes.processRoute.archiveSuccess'))
    getList()
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.processRoute.archiveFailed'))
    }
  }
}

const handleDelete = async (row: ProcessRoute) => {
  try {
    await ElMessageBox.confirm(t('mes.processRoute.confirmDelete'), t('common.warning'), {
      type: 'warning'
    })
    await deleteProcessRouteAPI(row.id!)
    ElMessage.success(t('mes.processRoute.deleteSuccess'))
    getList()
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') {
      ElMessage.error(t('mes.processRoute.deleteFailed'))
    }
  }
}

const getStatusType = (status?: string): 'success' | 'warning' | 'danger' | 'info' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | undefined> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    ARCHIVED: 'info'
  }
  return map[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const map: Record<string, string> = {
    DRAFT: t('mes.processRoute.statusDraft'),
    ACTIVE: t('mes.processRoute.statusActive'),
    ARCHIVED: t('mes.processRoute.statusArchived')
  }
  return map[status || ''] || status
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.process-route-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.toolbar {
  margin-bottom: 15px;
}
</style>
