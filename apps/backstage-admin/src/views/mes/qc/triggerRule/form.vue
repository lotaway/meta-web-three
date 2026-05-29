<template>
  <div class="trigger-rule-form">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.triggerRule.edit') : t('mes.qc.triggerRule.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="formData" :rules="rules" label-width="180px">
        <el-form-item :label="t('mes.qc.triggerRule.ruleCode')" prop="ruleCode">
          <el-input v-model="formData.ruleCode" :disabled="isEdit" :placeholder="t('mes.qc.triggerRule.ruleCodePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.ruleName')" prop="ruleName">
          <el-input v-model="formData.ruleName" :placeholder="t('mes.qc.triggerRule.ruleNamePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.triggerType')" prop="triggerType">
          <el-select v-model="formData.triggerType" :placeholder="t('mes.qc.triggerRule.triggerTypePlaceholder')">
            <el-option :label="t('mes.qc.triggerRule.triggerTypeAuto')" value="AUTO" />
            <el-option :label="t('mes.qc.triggerRule.triggerTypeManual')" value="MANUAL" />
            <el-option :label="t('mes.qc.triggerRule.triggerTypeScheduled')" value="SCHEDULED" />
            <el-option :label="t('mes.qc.triggerRule.triggerTypeEventBased')" value="EVENT_BASED" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.targetEntity')" prop="targetEntity">
          <el-input v-model="formData.targetEntity" :placeholder="t('mes.qc.triggerRule.targetEntityPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.conditionExpression')" prop="conditionExpression">
          <el-input v-model="formData.conditionExpression" type="textarea" :rows="2" :placeholder="t('mes.qc.triggerRule.conditionExpressionPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.actionType')" prop="actionType">
          <el-input v-model="formData.actionType" :placeholder="t('mes.qc.triggerRule.actionTypePlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.actionConfig')" prop="actionConfig">
          <el-input v-model="formData.actionConfig" type="textarea" :rows="2" :placeholder="t('mes.qc.triggerRule.actionConfigPlaceholder')" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.priority')" prop="priority">
          <el-input-number v-model="formData.priority" :min="0" :max="100" />
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.status')" prop="status">
          <el-select v-model="formData.status" :placeholder="t('mes.qc.triggerRule.statusPlaceholder')">
            <el-option :label="t('mes.qc.triggerRule.statusActive')" value="ACTIVE" />
            <el-option :label="t('mes.qc.triggerRule.statusInactive')" value="INACTIVE" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.qc.triggerRule.description')" prop="description">
          <el-input v-model="formData.description" type="textarea" :rows="3" :placeholder="t('mes.qc.triggerRule.descriptionPlaceholder')" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('mes.qc.triggerRule.save') }}</el-button>
          <el-button @click="handleCancel">{{ t('mes.qc.triggerRule.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import type { FormInstance } from 'element-plus'
import type { QcTriggerRule, TriggerType } from '@/apis/qc'
import {
  getTriggerRuleByIdAPI,
  createTriggerRuleAPI,
  updateTriggerRuleAPI
} from '@/apis/qc'

const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const formData = reactive<QcTriggerRule>({
  ruleCode: '',
  ruleName: '',
  triggerType: 'MANUAL' as TriggerType,
  targetEntity: '',
  conditionExpression: '',
  actionType: '',
  actionConfig: '',
  priority: 0,
  status: 'INACTIVE',
  description: ''
})

const { t } = useI18n()

const rules = {
  ruleCode: [
    { required: true, message: t('mes.qc.triggerRule.ruleCodePlaceholder'), trigger: 'blur' }
  ],
  ruleName: [
    { required: true, message: t('mes.qc.triggerRule.ruleNamePlaceholder'), trigger: 'blur' }
  ],
  triggerType: [
    { required: true, message: t('mes.qc.triggerRule.triggerTypePlaceholder'), trigger: 'change' }
  ]
}

const loadData = async () => {
  if (!route.query.id) return
  
  try {
    const res = await getTriggerRuleByIdAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(formData, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.qc.triggerRule.loadFailed'))
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    submitting.value = true
    try {
      if (isEdit.value) {
        await updateTriggerRuleAPI(Number(route.query.id), formData)
        ElMessage.success(t('mes.qc.triggerRule.updateSuccess'))
      } else {
        await createTriggerRuleAPI(formData)
        ElMessage.success(t('mes.qc.triggerRule.createSuccess'))
      }
      router.push('/mes/triggerRule')
    } catch (error) {
      ElMessage.error(t('mes.qc.triggerRule.deleteFailed'))
    } finally {
      submitting.value = false
    }
  })
}

const handleCancel = () => {
  router.back()
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.trigger-rule-form {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: 600;
}
</style>