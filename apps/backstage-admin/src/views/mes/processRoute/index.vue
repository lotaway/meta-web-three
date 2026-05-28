<template>
  <div class="process-route-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item label="路线编码">
          <el-input v-model="queryParams.routeCode" placeholder="请输入路线编码" clearable />
        </el-form-item>
        <el-form-item label="路线名称">
          <el-input v-model="queryParams.routeName" placeholder="请输入路线名称" clearable />
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="queryParams.status" placeholder="请选择状态" clearable>
            <el-option label="草稿" value="DRAFT" />
            <el-option label="已激活" value="ACTIVE" />
            <el-option label="已归档" value="ARCHIVED" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">查询</el-button>
          <el-button @click="resetQuery">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="toolbar">
        <el-button type="primary" @click="handleAdd">新增工艺路线</el-button>
      </div>

      <el-table :data="routeList" v-loading="loading" border stripe>
        <el-table-column label="ID" prop="id" width="80" />
        <el-table-column label="路线编码" prop="routeCode" width="150" />
        <el-table-column label="路线名称" prop="routeName" width="180" />
        <el-table-column label="产品编码" prop="productCode" width="150" />
        <el-table-column label="版本" prop="version" width="80" />
        <el-table-column label="状态" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="工序数量" width="100">
          <template #default="{ row }">
            {{ row.steps?.length || 0 }}
          </template>
        </el-table-column>
        <el-table-column label="创建时间" prop="createdAt" width="180" />
        <el-table-column label="操作" fixed="right" width="240">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">查看</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">编辑</el-button>
            <el-button link type="warning" size="small" @click="handleActivate(row)" v-if="row.status === 'DRAFT'">激活</el-button>
            <el-button link type="info" size="small" @click="handleArchive(row)" v-if="row.status === 'ACTIVE'">归档</el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">删除</el-button>
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
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ProcessRoute } from '@/apis/processRoute'
import { 
  getProcessRouteListAPI, 
  deleteProcessRouteAPI,
  activateProcessRouteAPI,
  archiveProcessRouteAPI 
} from '@/apis/processRoute'

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
    const data = await getProcessRouteListAPI(queryParams.status || undefined)
    routeList.value = data || []
    total.value = data?.length || 0
  } catch (error) {
    console.error('获取工艺路线列表失败:', error)
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
    await ElMessageBox.confirm('确定要激活该工艺路线吗？', '提示', {
      type: 'warning'
    })
    await activateProcessRouteAPI(row.id!)
    ElMessage.success('激活成功')
    getList()
  } catch (error) {
    // 用户取消或激活失败
  }
}

const handleArchive = async (row: ProcessRoute) => {
  try {
    await ElMessageBox.confirm('确定要归档该工艺路线吗？', '提示', {
      type: 'warning'
    })
    await archiveProcessRouteAPI(row.id!)
    ElMessage.success('归档成功')
    getList()
  } catch (error) {
    // 用户取消或归档失败
  }
}

const handleDelete = async (row: ProcessRoute) => {
  try {
    await ElMessageBox.confirm('确定要删除该工艺路线吗？', '提示', {
      type: 'warning'
    })
    await deleteProcessRouteAPI(row.id!)
    ElMessage.success('删除成功')
    getList()
  } catch (error) {
    // 用户取消或删除失败
  }
}

const getStatusType = (status?: string) => {
  const map: Record<string, string> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    ARCHIVED: ''
  }
  return map[status || ''] || 'info'
}

const getStatusText = (status?: string) => {
  const map: Record<string, string> = {
    DRAFT: '草稿',
    ACTIVE: '已激活',
    ARCHIVED: '已归档'
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
