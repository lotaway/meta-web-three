<template>
  <div class="pokayoke-rule-form-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? '编辑规则' : '新增规则' }}</span>
          <el-button @click="handleBack">返回</el-button>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="规则编码" prop="ruleCode">
              <el-input v-model="form.ruleCode" placeholder="请输入规则编码" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="规则名称" prop="ruleName">
              <el-input v-model="form.ruleName" placeholder="请输入规则名称" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="规则类型" prop="ruleType">
              <el-select v-model="form.ruleType" placeholder="请选择规则类型">
                <el-option label="物料检查" value="MATERIAL_CHECK" />
                <el-option label="顺序检查" value="SEQUENCE_CHECK" />
                <el-option label="参数检查" value="PARAMETER_CHECK" />
                <el-option label="工位检查" value="STATION_CHECK" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="优先级">
              <el-input-number v-model="form.priority" :min="0" :max="100" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="工位">
              <el-input v-model="form.workstationId" placeholder="请输入工位ID" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="错误提示">
              <el-input v-model="form.errorMessage" placeholder="请输入错误提示" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item label="条件表达式">
          <el-input 
            v-model="form.conditionExpression" 
            type="textarea" 
            :rows="4"
            placeholder="例如: material.code.startsWith('MAT-')" 
          />
          <div class="form-tip">
            支持的变量: material.code, material.quantity, sequence.current, sequence.next, 
            parameter.name, parameter.value, workstation.id
          </div>
        </el-form-item>

        <el-form-item label="动作类型">
          <el-select v-model="form.actionType" placeholder="请选择动作类型">
            <el-option label="阻止操作" value="BLOCK" />
            <el-option label="警告" value="WARNING" />
            <el-option label="记录日志" value="LOG" />
          </el-select>
        </el-form-item>

        <el-form-item label="动作配置">
          <el-input 
            v-model="form.actionConfig" 
            type="textarea" 
            :rows="3"
            placeholder="JSON 格式的动作配置" 
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">保存</el-button>
          <el-button @click="handleBack">取消</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { usePokayokeRuleStore } from '@/stores/pokayokeRule'
import type { FormInstance } from 'element-plus'

const route = useRoute()
const router = useRouter()
const ruleStore = usePokayokeRuleStore()

const formRef = ref<FormInstance>()
const submitting = ref(false)

const isEdit = computed(() => !!route.query.id)

const form = reactive({
  ruleCode: '',
  ruleName: '',
  ruleType: 'MATERIAL_CHECK' as string,
  priority: 0,
  workstationId: '',
  conditionExpression: '',
  actionType: 'BLOCK',
  actionConfig: '',
  errorMessage: '',
})

const rules = {
  ruleCode: [{ required: true, message: '请输入规则编码', trigger: 'blur' }],
  ruleName: [{ required: true, message: '请输入规则名称', trigger: 'blur' }],
  ruleType: [{ required: true, message: '请选择规则类型', trigger: 'change' }],
}

onMounted(async () => {
  if (isEdit.value) {
    const id = Number(route.query.id)
    const data = await ruleStore.fetchRuleById(id)
    if (data) {
      Object.assign(form, {
        ruleCode: data.ruleCode,
        ruleName: data.ruleName,
        ruleType: data.ruleType,
        priority: data.priority || 0,
        workstationId: data.workstationId || '',
        conditionExpression: data.conditionExpression || '',
        actionType: data.actionType || 'BLOCK',
        actionConfig: data.actionConfig || '',
        errorMessage: data.errorMessage || '',
      })
    }
  }
})

async function handleSubmit() {
  if (!formRef.value) return
  
  await formRef.value.validate()
  
  submitting.value = true
  try {
    if (isEdit.value) {
      await ruleStore.updateRule(Number(route.query.id), form)
      ElMessage.success('更新成功')
    } else {
      await ruleStore.createRule(form)
      ElMessage.success('创建成功')
    }
    router.push('/mes/pokayoke')
  } catch (error) {
    ElMessage.error('操作失败')
  } finally {
    submitting.value = false
  }
}

function handleBack() {
  router.back()
}
</script>

<style scoped>
.pokayoke-rule-form-container {
  padding: 16px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}
</style>