<template>
  <div class="project-container">
    <div class="header">
      <h2>{{ t('project.management') }}</h2>
      <el-button type="primary" @click="handleCreate">{{ t('project.create') }}</el-button>
    </div>

    <el-form :inline="true" :model="queryParams" class="search-form">
      <el-form-item :label="t('project.keyword')">
        <el-input v-model="queryParams.keyword" :placeholder="t('project.codeName')" clearable />
      </el-form-item>
      <el-form-item :label="t('mes.processRoute.status')">
        <el-select v-model="queryParams.status" :placeholder="t('common.selectPlaceholder')" clearable>
          <el-option :label="t('mes.processRoute.statusDraft')" value="DRAFT" />
          <el-option :label="t('mes.productionTask.statusIN_PROGRESS')" value="ACTIVE" />
          <el-option :label="t('mes.workOrder.statusPAUSED')" value="SUSPENDED" />
          <el-option :label="t('mes.equipment.maintenancePlan.statusCompleted')" value="COMPLETED" />
          <el-option :label="t('mes.productionTask.statusCANCELLED')" value="CANCELLED" />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="handleQuery">{{ t('common.query') }}</el-button>
        <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="projectList" border v-loading="loading">
      <el-table-column prop="projectCode" :label="t('project.code')" width="120" />
      <el-table-column prop="projectName" :label="t('project.name')" min-width="150" />
      <el-table-column prop="departmentName" :label="t('project.department')" width="120" />
      <el-table-column prop="managerName" :label="t('project.manager')" width="100" />
      <el-table-column prop="budgetAmount" :label="t('project.budget')" width="120" />
      <el-table-column prop="usedAmount" :label="t('project.used')" width="120" />
      <el-table-column prop="progress" :label="t('project.progress')" width="100">
        <template #default="{ row }">
          <el-progress :percentage="row.progress" :stroke-width="10" />
        </template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="startDate" label="开始日期" width="120" />
      <el-table-column prop="endDate" label="结束日期" width="120" />
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" @click="handleView(row as Project)">{{ t('product.view') }}</el-button>
          <el-button link type="primary" @click="handleEdit(row as Project)">{{ t('common.edit') }}</el-button>
          <el-button link type="danger" @click="handleDelete(row as Project)">{{ t('common.delete') }}</el-button>
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

    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="600px">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="100px">
        <el-form-item label="项目编码" prop="projectCode">
          <el-input v-model="form.projectCode" :disabled="isEdit" />
        </el-form-item>
        <el-form-item label="项目名称" prop="projectName">
          <el-input v-model="form.projectName" />
        </el-form-item>
        <el-form-item label="所属部门" prop="departmentName">
          <el-input v-model="form.departmentName" />
        </el-form-item>
        <el-form-item label="项目经理" prop="managerName">
          <el-input v-model="form.managerName" />
        </el-form-item>
        <el-form-item label="预算金额" prop="budgetAmount">
          <el-input-number v-model="form.budgetAmount" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item label="开始日期" prop="startDate">
          <el-date-picker v-model="form.startDate" type="date" value-format="YYYY-MM-DD" />
        </el-form-item>
        <el-form-item label="结束日期" prop="endDate">
          <el-date-picker v-model="form.endDate" type="date" value-format="YYYY-MM-DD" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input v-model="form.description" type="textarea" :rows="3" />
        </el-form-item>
        <el-form-item label="备注" prop="remark">
          <el-input v-model="form.remark" type="textarea" :rows="2" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="submitForm">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance } from 'element-plus'
import { getProjectList, createProject, updateProject, deleteProject } from '@/apis/project'
import type { Project, ProjectCreateCommand, ProjectUpdateCommand } from '@/apis/project'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const loading = ref(false)
const projectList = ref<Project[]>([])
const total = ref(0)
const dialogVisible = ref(false)
const dialogTitle = ref('')
const isEdit = ref(false)
const formRef = ref<FormInstance>()

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
  keyword: '',
  status: ''
})

const form = reactive<ProjectCreateCommand & { id?: number }>({
  projectCode: '',
  projectName: '',
  description: '',
  departmentId: 0,
  departmentName: '',
  managerId: 0,
  managerName: '',
  startDate: '',
  endDate: '',
  budgetAmount: 0,
  currency: 'CNY',
  remark: ''
})

const rules = {
  projectCode: [{ required: true, message: '请输入项目编码', trigger: 'blur' }],
  projectName: [{ required: true, message: '请输入项目名称', trigger: 'blur' }],
  departmentName: [{ required: true, message: '请输入所属部门', trigger: 'blur' }],
  managerName: [{ required: true, message: '请输入项目经理', trigger: 'blur' }]
}

const getStatusType = (status: string): 'success' | 'warning' | 'danger' | 'info' | 'primary' => {
  const map: Record<string, 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    DRAFT: 'info',
    ACTIVE: 'success',
    SUSPENDED: 'warning',
    COMPLETED: 'primary',
    CANCELLED: 'danger'
  }
  return map[status] || 'info'
}

const getStatusText = (status: string) => {
  const map: Record<string, string> = {
    DRAFT: '草稿',
    ACTIVE: '进行中',
    SUSPENDED: '已暂停',
    COMPLETED: '已完成',
    CANCELLED: '已取消'
  }
  return map[status] || status
}

const getList = async () => {
  loading.value = true
  try {
    const res = await getProjectList(queryParams)
    if (res.data.code === 200) {
      projectList.value = res.data.data.list
      total.value = res.data.data.total
    }
  } finally {
    loading.value = false
  }
}

const handleQuery = () => {
  queryParams.pageNum = 1
  getList()
}

const handleReset = () => {
  queryParams.keyword = ''
  queryParams.status = ''
  handleQuery()
}

const handleCreate = () => {
  dialogTitle.value = '新建项目'
  isEdit.value = false
  Object.assign(form, {
    projectCode: '',
    projectName: '',
    description: '',
    departmentId: 0,
    departmentName: '',
    managerId: 0,
    managerName: '',
    startDate: '',
    endDate: '',
    budgetAmount: 0,
    currency: 'CNY',
    remark: ''
  })
  dialogVisible.value = true
}

const handleEdit = (row: Project) => {
  dialogTitle.value = '编辑项目'
  isEdit.value = true
  Object.assign(form, {
    id: row.id,
    projectCode: row.projectCode,
    projectName: row.projectName,
    description: row.description,
    departmentId: row.departmentId,
    departmentName: row.departmentName,
    managerId: row.managerId,
    managerName: row.managerName,
    startDate: row.startDate,
    endDate: row.endDate,
    budgetAmount: row.budgetAmount,
    remark: row.remark || ''
  })
  dialogVisible.value = true
}

const handleView = (row: Project) => {
  // TODO: Navigate to detail page
  ElMessage.info(t('project.viewDetail') + row.projectName)
}

const handleDelete = async (row: Project) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmDeleteProject'), t('dialog.title'), { type: 'warning' })
    const res = await deleteProject(row.id)
    if (res.data.code === 200) {
      ElMessage.success(t('common.deleteSuccess'))
      getList()
    }
  } catch {
    // cancel
  }
}

const submitForm = async () => {
  if (!formRef.value) return
  await formRef.value.validate()
  try {
    if (isEdit.value) {
      const res = await updateProject(form as ProjectUpdateCommand)
      if (res.data.code === 200) {
        ElMessage.success(t('common.updateSuccess'))
        dialogVisible.value = false
        getList()
      }
    } else {
      const res = await createProject(form)
      if (res.data.code === 200) {
        ElMessage.success(t('common.createSuccess'))
        dialogVisible.value = false
        getList()
      }
    }
  } catch {
    // error
  }
}

onMounted(() => {
  getList()
})
</script>

<style scoped>
.project-container {
  padding: 20px;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.search-form {
  margin-bottom: 20px;
}
</style>