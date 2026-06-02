<template>
  <div class="recommendation-rule-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="demo-form-inline">
        <el-form-item label="Scene">
          <el-select v-model="queryParams.scene" placeholder="Select Scene" clearable>
            <el-option label="Home" value="HOME" />
            <el-option label="Product Detail" value="PRODUCT_DETAIL" />
            <el-option label="Cart" value="CART" />
            <el-option label="Order Complete" value="ORDER_COMPLETE" />
          </el-select>
        </el-form-item>
        <el-form-item label="Type">
          <el-select v-model="queryParams.type" placeholder="Select Type" clearable>
            <el-option label="Collaborative" value="COLLABORATIVE" />
            <el-option label="Content-Based" value="CONTENT_BASED" />
            <el-option label="Hybrid" value="HYBRID" />
            <el-option label="Popularity" value="POPULARITY" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">Query</el-button>
          <el-button @click="resetQuery">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="table-operations">
        <el-button type="primary" @click="handleCreate">Create Rule</el-button>
      </div>

      <el-table :data="ruleList" border style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="ruleName" label="Rule Name" width="180" />
        <el-table-column prop="scene" label="Scene" width="150">
          <template #default="{ row }">
            <el-tag>{{ row.scene }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="type" label="Type" width="150">
          <template #default="{ row }">
            <el-tag :type="getTypeTagType(row.type)">{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="isActive" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.isActive ? 'success' : 'info'">
              {{ row.isActive ? 'Active' : 'Inactive' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="Created At" width="180" />
        <el-table-column label="Operations" fixed="right" width="150">
          <template #default="{ row }">
            <el-button v-if="!row.isActive" type="success" link @click="handleActivate(row)">
              Activate
            </el-button>
            <el-button type="danger" link @click="handleDelete(row)">Delete</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Create Rule Dialog -->
    <el-dialog v-model="createDialogVisible" title="Create Recommendation Rule" width="500px">
      <el-form ref="ruleFormRef" :model="ruleForm" :rules="rules" label-width="120px">
        <el-form-item label="Rule Name" prop="ruleName">
          <el-input v-model="ruleForm.ruleName" />
        </el-form-item>
        <el-form-item label="Scene" prop="scene">
          <el-select v-model="ruleForm.scene" placeholder="Select Scene">
            <el-option label="Home" value="HOME" />
            <el-option label="Product Detail" value="PRODUCT_DETAIL" />
            <el-option label="Cart" value="CART" />
            <el-option label="Order Complete" value="ORDER_COMPLETE" />
          </el-select>
        </el-form-item>
        <el-form-item label="Type" prop="type">
          <el-select v-model="ruleForm.type" placeholder="Select Type">
            <el-option label="Collaborative Filtering" value="COLLABORATIVE" />
            <el-option label="Content-Based" value="CONTENT_BASED" />
            <el-option label="Hybrid" value="HYBRID" />
            <el-option label="Popularity-Based" value="POPULARITY" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="submitForm">Confirm</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance } from 'element-plus'
import {
  getRulesBySceneAPI,
  createRuleAPI,
  activateRuleAPI,
  deleteRuleAPI,
  type RecommendationRule,
  type CreateRuleParams
} from '@/apis/recommendation'

const queryParams = reactive({
  scene: '',
  type: ''
})

const ruleList = ref<RecommendationRule[]>([])
const createDialogVisible = ref(false)
const ruleFormRef = ref<FormInstance>()

const ruleForm = reactive<CreateRuleParams>({
  ruleName: '',
  scene: '',
  type: ''
})

const rules = {
  ruleName: [{ required: true, message: 'Please input rule name', trigger: 'blur' }],
  scene: [{ required: true, message: 'Please select scene', trigger: 'change' }],
  type: [{ required: true, message: 'Please select type', trigger: 'change' }]
}

const getTypeTagType = (type: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    COLLABORATIVE: 'primary',
    CONTENT_BASED: 'success',
    HYBRID: 'warning',
    POPULARITY: 'info'
  }
  return typeMap[type] || 'info'
}

const handleQuery = async () => {
  try {
    const scene = queryParams.scene || 'HOME'
    const response = await getRulesBySceneAPI(scene)
    ruleList.value = response.data || []
  } catch (error) {
    ElMessage.error('Failed to load rules')
  }
}

const resetQuery = () => {
  queryParams.scene = ''
  queryParams.type = ''
  handleQuery()
}

const handleCreate = () => {
  ruleForm.ruleName = ''
  ruleForm.scene = ''
  ruleForm.type = ''
  createDialogVisible.value = true
}

const submitForm = async () => {
  if (!ruleFormRef.value) return
  await ruleFormRef.value.validate(async (valid) => {
    if (valid) {
      try {
        await createRuleAPI(ruleForm)
        ElMessage.success('Rule created successfully')
        createDialogVisible.value = false
        handleQuery()
      } catch (error) {
        ElMessage.error('Failed to create rule')
      }
    }
  })
}

const handleActivate = async (row: RecommendationRule) => {
  if (!row.id) {
    ElMessage.warning('Invalid rule ID')
    return
  }
  try {
    await activateRuleAPI(row.id)
    ElMessage.success('Rule activated successfully')
    handleQuery()
  } catch (error) {
    ElMessage.error('Failed to activate rule')
  }
}

const handleDelete = async (row: RecommendationRule) => {
  if (!row.id) {
    ElMessage.warning('Invalid rule ID')
    return
  }
  try {
    await ElMessageBox.confirm('Are you sure to delete this rule?', 'Warning', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await deleteRuleAPI(row.id)
    ElMessage.success('Rule deleted')
    handleQuery()
  } catch {
    // User cancelled or error
  }
}

onMounted(() => {
  handleQuery()
})
</script>

<style scoped>
.recommendation-rule-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.table-operations {
  margin-bottom: 15px;
}
</style>