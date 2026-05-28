<template>
  <div class="inspection-type-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="filter-form">
        <el-form-item label="检验类型编码">
          <el-input v-model="queryParams.typeCode" placeholder="请输入检验类型编码" clearable />
        </el-form-item>
        <el-form-item label="检验类型名称">
          <el-input v-model="queryParams.typeName" placeholder="请输入检验类型名称" clearable />
        </el-form-item>
        <el-form-item label="检验分类">
          <el-select v-model="queryParams.category" placeholder="请选择分类" clearable>
            <el-option label="来料检验" value="INCOMING" />
            <el-option label="工序检验" value="PROCESS" />
            <el-option label="最终检验" value="FINAL" />
            <el-option label="出货检验" value="OUTGOING" />
            <el-option label="自定义" value="CUSTOM" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="queryParams.status" placeholder="请选择状态" clearable>
            <el-option label="激活" value="ACTIVE" />
            <el-option label="停用" value="INACTIVE" />
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
        <el-button type="primary" @click="handleAdd">新增</el-button>
      </div>

      <el-table :data="typeList" v-loading="loading" border stripe>
        <el-table-column label="ID" prop="id" width="80" />
        <el-table-column label="检验类型编码" prop="typeCode" width="150" />
        <el-table-column label="检验类型名称" prop="typeName" width="180" />
        <el-table-column label="检验分类" prop="category" width="120">
          <template #default="{ row }">
            {{ getCategoryText(row.category) }}
          </template>
        </el-table-column>
        <el-table-column label="默认抽样方案" prop="defaultSamplingPlan" width="150" />
        <el-table-column label="默认AQL" prop="defaultAql" width="100" />
        <el-table-column label="需要证书" prop="requireCertificate" width="100">
          <template #default="{ row }">
            {{ row.requireCertificate ? '是' : '否' }}
          </template>
        </el-table-column>
        <el-table-column label="状态" prop="status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'">
              {{ row.status === 'ACTIVE' ? '激活' : '停用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="排序" prop="sortOrder" width="80" />
        <el-table-column label="创建时间" prop="createdAt" width="180" />
        <el-table-column label="操作" fixed="right" width="200">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">查看</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">编辑</el-button>
            <el-button link type="success" size="small" @click="handleActivate(row)" v-if="row.status === 'INACTIVE'">激活</el-button>
            <el-button link type="warning" size="small" @click="handleDeactivate(row)" v-if="row.status === 'ACTIVE'">停用</el-button>
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
import type { QcInspectionType, InspectionCategory } from '@/apis/qc'
import {
  getInspectionTypeListAPI,
  deleteInspectionTypeAPI,
  activateInspectionTypeAPI,
  deactivateInspectionTypeAPI
} from '@/apis/qc'

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
    INCOMING: '来料检验',
    PROCESS: '工序检验',
    FINAL: '最终检验',
    OUTGOING: '出货检验',
    CUSTOM: '自定义'
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
    ElMessage.error('获取检验类型列表失败')
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
    await ElMessageBox.confirm('确定要激活该检验类型吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await activateInspectionTypeAPI(row.id!)
    ElMessage.success('激活成功')
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('激活失败')
    }
  }
}

const handleDeactivate = async (row: QcInspectionType) => {
  try {
    await ElMessageBox.confirm('确定要停用该检验类型吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await deactivateInspectionTypeAPI(row.id!)
    ElMessage.success('停用成功')
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('停用失败')
    }
  }
}

const handleDelete = async (row: QcInspectionType) => {
  try {
    await ElMessageBox.confirm('确定要删除该检验类型吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await deleteInspectionTypeAPI(row.id!)
    ElMessage.success('删除成功')
    getList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
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
