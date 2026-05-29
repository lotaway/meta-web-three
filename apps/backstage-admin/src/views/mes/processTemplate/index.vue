<template>
  <div class="process-template-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="filterForm" class="filter-form">
        <el-form-item :label="t('mes.processTemplate.status')">
          <el-select
            v-model="filterForm.status"
            :placeholder="t('mes.processTemplate.statusPlaceholder')"
            clearable
            @change="handleFilterChange"
          >
            <el-option label="" value="" />
            <el-option :label="t('mes.processTemplate.statusDraft')" value="DRAFT" />
            <el-option :label="t('mes.processTemplate.statusPublished')" value="PUBLISHED" />
            <el-option :label="t('mes.processTemplate.statusArchived')" value="ARCHIVED" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleFilterChange">
            {{ t('mes.processTemplate.search') }}
          </el-button>
          <el-button @click="handleReset">{{ t('mes.processTemplate.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.processTemplate.title') }}</span>
          <el-button type="primary" @click="handleAdd">
            {{ t('mes.processTemplate.add') }}
          </el-button>
        </div>
      </template>

      <el-table :data="tableData" v-loading="loading" border stripe>
        <el-table-column prop="id" :label="t('mes.processTemplate.id')" width="80" />
        <el-table-column prop="templateCode" :label="t('mes.processTemplate.templateCode')" min-width="150" />
        <el-table-column prop="templateName" :label="t('mes.processTemplate.templateName')" min-width="150" />
        <el-table-column prop="description" :label="t('mes.processTemplate.description')" min-width="200" show-overflow-tooltip />
        <el-table-column prop="version" :label="t('mes.processTemplate.version')" width="100" />
        <el-table-column prop="status" :label="t('mes.processTemplate.status')" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" :label="t('mes.processTemplate.createdAt')" width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column :label="t('mes.processTemplate.actions')" width="280" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">
              {{ t('mes.processTemplate.view') }}
            </el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">
              {{ t('mes.processTemplate.edit') }}
            </el-button>
            <el-button link type="success" size="small" @click="handlePublish(row)" v-if="row.status === 'DRAFT'">
              {{ t('mes.processTemplate.publish') }}
            </el-button>
            <el-button link type="warning" size="small" @click="handleArchive(row)" v-if="row.status === 'PUBLISHED'">
              {{ t('mes.processTemplate.archive') }}
            </el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">
              {{ t('mes.processTemplate.delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-container">
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :total="pagination.total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listProcessTemplatesAPI,
  deleteProcessTemplateAPI,
  publishProcessTemplateAPI,
  archiveProcessTemplateAPI,
} from '@/apis/processFlow'
import type { ProcessFlowTemplate } from '@/apis/processFlow'

const { t } = useI18n()
const router = useRouter()

const loading = ref(false)
const tableData = ref<ProcessFlowTemplate[]>([])

const filterForm = reactive({
  status: '',
})

const pagination = reactive({
  page: 1,
  pageSize: 10,
  total: 0,
})

const getStatusType = (status?: string) => {
  const statusMap: Record<string, string> = {
    DRAFT: 'info',
    PUBLISHED: 'success',
    ARCHIVED: 'warning',
  }
  return statusMap[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const textMap: Record<string, string> = {
    DRAFT: t('mes.processTemplate.statusDraft'),
    PUBLISHED: t('mes.processTemplate.statusPublished'),
    ARCHIVED: t('mes.processTemplate.statusArchived'),
  }
  return textMap[status || ''] || status
}

const formatDateTime = (dateStr?: string) => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

const fetchData = async () => {
  loading.value = true
  try {
    const res = await listProcessTemplatesAPI(filterForm.status || undefined)
    tableData.value = res.data || []
    pagination.total = tableData.value.length
  } catch (error) {
    ElMessage.error(t('mes.processTemplate.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleFilterChange = () => {
  pagination.page = 1
  fetchData()
}

const handleReset = () => {
  filterForm.status = ''
  handleFilterChange()
}

const handleAdd = () => {
  router.push('/mes/process-template/form')
}

const handleEdit = (row: ProcessFlowTemplate) => {
  router.push(`/mes/process-template/form?id=${row.id}`)
}

const handleView = (row: ProcessFlowTemplate) => {
  router.push(`/mes/process-template/detail?id=${row.id}`)
}

const handlePublish = async (row: ProcessFlowTemplate) => {
  try {
    await ElMessageBox.confirm(
      t('mes.processTemplate.confirmPublish'),
      t('mes.processTemplate.warning'),
      { confirmButtonText: t('mes.processTemplate.confirm'), cancelButtonText: t('mes.processTemplate.cancel'), type: 'warning' }
    )
    await publishProcessTemplateAPI(row.id!, 1)
    ElMessage.success(t('mes.processTemplate.publishSuccess'))
    fetchData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processTemplate.publishFailed'))
    }
  }
}

const handleArchive = async (row: ProcessFlowTemplate) => {
  try {
    await ElMessageBox.confirm(
      t('mes.processTemplate.confirmArchive'),
      t('mes.processTemplate.warning'),
      { confirmButtonText: t('mes.processTemplate.confirm'), cancelButtonText: t('mes.processTemplate.cancel'), type: 'warning' }
    )
    await archiveProcessTemplateAPI(row.id!, 1)
    ElMessage.success(t('mes.processTemplate.archiveSuccess'))
    fetchData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processTemplate.archiveFailed'))
    }
  }
}

const handleDelete = async (row: ProcessFlowTemplate) => {
  try {
    await ElMessageBox.confirm(
      t('mes.processTemplate.confirmDelete'),
      t('mes.processTemplate.warning'),
      { confirmButtonText: t('mes.processTemplate.confirm'), cancelButtonText: t('mes.processTemplate.cancel'), type: 'warning' }
    )
    await deleteProcessTemplateAPI(row.id!)
    ElMessage.success(t('mes.processTemplate.deleteSuccess'))
    fetchData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(t('mes.processTemplate.deleteFailed'))
    }
  }
}

const handleSizeChange = (size: number) => {
  pagination.pageSize = size
  fetchData()
}

const handlePageChange = (page: number) => {
  pagination.page = page
  fetchData()
}

onMounted(() => {
  fetchData()
})
</script>

<style scoped>
.process-template-container {
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

.table-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
}
</style>