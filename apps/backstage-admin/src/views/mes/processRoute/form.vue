<template>
  <div class="process-route-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? '编辑工艺路线' : '新增工艺路线' }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="路线编码" prop="routeCode">
              <el-input v-model="form.routeCode" placeholder="请输入路线编码" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="路线名称" prop="routeName">
              <el-input v-model="form.routeName" placeholder="请输入路线名称" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="产品编码" prop="productCode">
              <el-input v-model="form.productCode" placeholder="请输入产品编码" />
            </el-form-item>
          </el-col>
          <el-col :span="12" v-if="isEdit">
            <el-form-item label="版本">
              <el-input :value="form.version" disabled />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">工序步骤</el-divider>

        <div class="steps-container">
          <el-table :data="form.steps" border stripe>
            <el-table-column label="序号" prop="stepNo" width="80" align="center">
              <template #default="{ $index }">
                {{ $index + 1 }}
              </template>
            </el-table-column>
            <el-table-column label="工序编码" prop="processCode" min-width="120">
              <template #default="{ row }">
                <el-input v-model="row.processCode" placeholder="工序编码" size="small" />
              </template>
            </el-table-column>
            <el-table-column label="工序名称" prop="processName" min-width="120">
              <template #default="{ row }">
                <el-input v-model="row.processName" placeholder="工序名称" size="small" />
              </template>
            </el-table-column>
            <el-table-column label="工位ID" prop="workstationId" width="120">
              <template #default="{ row }">
                <el-input v-model="row.workstationId" placeholder="工位ID" size="small" />
              </template>
            </el-table-column>
            <el-table-column label="标准工时(分钟)" prop="standardTime" width="140">
              <template #default="{ row }">
                <el-input-number v-model="row.standardTime" :min="0" :step="1" size="small" controls-position="right" />
              </template>
            </el-table-column>
            <el-table-column label="质检工序" prop="qualityCheckpoint" width="120">
              <template #default="{ row }">
                <el-switch v-model="row.qualityCheckpoint" active-value="YES" inactive-value="NO" />
              </template>
            </el-table-column>
            <el-table-column label="前驱工序" prop="predecessorStepNo" width="100">
              <template #default="{ row, $index }">
                <el-select v-model="row.predecessorStepNo" placeholder="选择" clearable size="small">
                  <el-option v-for="(step, idx) in getAvailablePredecessors($index)" :key="idx" :label="step.processName || `步骤${idx+1}`" :value="step.stepNo" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="80" align="center">
              <template #default="{ $index }">
                <el-button link type="danger" size="small" @click="removeStep($index)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>

          <el-button type="primary" plain size="small" @click="addStep" style="margin-top: 10px">
            <el-icon><Plus /></el-icon>
            添加工序
          </el-button>
        </div>

        <el-divider />

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">提交</el-button>
          <el-button @click="handleCancel">取消</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import type { ProcessRoute, ProcessStep, CreateProcessRouteRequest, UpdateProcessRouteRequest } from '@/apis/processRoute'
import { 
  getProcessRouteByIdAPI, 
  createProcessRouteAPI, 
  updateProcessRouteAPI,
  validateProcessRouteAPI 
} from '@/apis/processRoute'

const route = useRoute()
const router = useRouter()
const formRef = ref<FormInstance>()
const submitting = ref(false)

const isEdit = computed(() => !!route.query.id)
const routeId = computed(() => Number(route.query.id) || 0)

const form = reactive<{
  routeCode: string
  routeName: string
  productCode: string
  version?: number
  status?: string
  steps: ProcessStep[]
}>({
  routeCode: '',
  routeName: '',
  productCode: '',
  version: undefined,
  status: undefined,
  steps: []
})

const rules: FormRules = {
  routeCode: [{ required: true, message: '请输入路线编码', trigger: 'blur' }],
  routeName: [{ required: true, message: '请输入路线名称', trigger: 'blur' }],
  productCode: [{ required: true, message: '请输入产品编码', trigger: 'blur' }],
}

const addStep = () => {
  const stepNo = form.steps.length + 1
  form.steps.push({
    stepNo,
    processCode: '',
    processName: '',
    workstationId: undefined,
    standardTime: undefined,
    qualityCheckpoint: 'NO',
    predecessorStepNo: undefined,
    successorStepNo: undefined
  })
}

const removeStep = (index: number) => {
  form.steps.splice(index, 1)
  // 重新编号
  form.steps.forEach((step, idx) => {
    step.stepNo = idx + 1
  })
}

const getAvailablePredecessors = (currentIndex: number) => {
  return form.steps.filter((_, idx) => idx < currentIndex)
}

const loadData = async () => {
  if (!isEdit.value) return
  
  try {
    const data = await getProcessRouteByIdAPI(routeId.value)
    if (data) {
      form.routeCode = data.routeCode
      form.routeName = data.routeName
      form.productCode = data.productCode
      form.version = data.version
      form.status = data.status
      form.steps = data.steps ? [...data.steps] : []
    }
  } catch (error) {
    ElMessage.error('加载数据失败')
    console.error(error)
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    if (form.steps.length === 0) {
      ElMessage.warning('请至少添加一个工序')
      return
    }

    submitting.value = true
    try {
      if (isEdit.value) {
        const data: UpdateProcessRouteRequest = {
          routeName: form.routeName,
          productCode: form.productCode,
          steps: form.steps
        }
        await updateProcessRouteAPI(routeId.value, data)
        ElMessage.success('更新成功')
      } else {
        const data: CreateProcessRouteRequest = {
          routeCode: form.routeCode,
          routeName: form.routeName,
          productCode: form.productCode,
          steps: form.steps
        }
        await createProcessRouteAPI(data)
        ElMessage.success('创建成功')
      }
      
      // 可选：保存后验证
      // await validateProcessRouteAPI(...)
      
      router.push('/mes/process-route')
    } catch (error) {
      console.error('提交失败:', error)
    } finally {
      submitting.value = false
    }
  })
}

const handleCancel = () => {
  router.push('/mes/process-route')
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.process-route-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}

.steps-container {
  margin: 10px 0;
}
</style>