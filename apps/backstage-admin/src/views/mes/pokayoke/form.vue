<template>
  <div class="pokayoke-rule-form-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('pokayoke.edit') : t('pokayoke.add') }}</span>
          <el-button @click="handleBack">{{ t('pokayoke.back') }}</el-button>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('pokayoke.ruleCode')" prop="ruleCode">
              <el-input v-model="form.ruleCode" :placeholder="t('pokayoke.ruleCodePlaceholder')" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('pokayoke.ruleName')" prop="ruleName">
              <el-input v-model="form.ruleName" :placeholder="t('pokayoke.ruleNamePlaceholder')" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('pokayoke.ruleType')" prop="ruleType">
              <el-select v-model="form.ruleType" :placeholder="t('pokayoke.ruleTypePlaceholder')">
                <el-option :label="t('pokayoke.typeMaterialCheck')" value="MATERIAL_CHECK" />
                <el-option :label="t('pokayoke.typeSequenceCheck')" value="SEQUENCE_CHECK" />
                <el-option :label="t('pokayoke.typeParameterCheck')" value="PARAMETER_CHECK" />
                <el-option :label="t('pokayoke.typeStationCheck')" value="STATION_CHECK" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('pokayoke.priority')">
              <el-input-number v-model="form.priority" :min="0" :max="100" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('pokayoke.workstation')">
              <el-input v-model="form.workstationId" :placeholder="t('pokayoke.workstationIdPlaceholder')" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('pokayoke.errorMessage')">
              <el-input v-model="form.errorMessage" :placeholder="t('pokayoke.errorMessagePlaceholder')" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item :label="t('pokayoke.conditionExpression')">
          <el-input 
            v-model="form.conditionExpression" 
            type="textarea" 
            :rows="4"
            :placeholder="t('pokayoke.conditionExpressionPlaceholder')" 
          />
          <div class="form-tip">
            {{ t('pokayoke.conditionExpressionTip') }}
          </div>
        </el-form-item>

        <el-form-item :label="t('pokayoke.actionType')">
          <el-select v-model="form.actionType" :placeholder="t('pokayoke.actionTypePlaceholder')">
            <el-option :label="t('pokayoke.actionBlock')" value="BLOCK" />
            <el-option :label="t('pokayoke.actionWarning')" value="WARNING" />
            <el-option :label="t('pokayoke.actionLog')" value="LOG" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('pokayoke.actionConfig')">
          <el-input 
            v-model="form.actionConfig" 
            type="textarea" 
            :rows="3"
            :placeholder="t('pokayoke.actionConfigPlaceholder')" 
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('pokayoke.save') }}</el-button>
          <el-button @click="handleBack">{{ t('pokayoke.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useI18n } from 'vue-i18n'
import { usePokayokeRuleStore } from '@/stores/pokayokeRule'
import type { FormInstance } from 'element-plus'
import type { RuleType } from '@/apis/pokayokeRule'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const ruleStore = usePokayokeRuleStore()

const formRef = ref<FormInstance>()
const submitting = ref(false)

const isEdit = computed(() => !!route.query.id)

const form = reactive<{
  ruleCode: string
  ruleName: string
  ruleType: RuleType
  priority: number
  workstationId: string
  conditionExpression: string
  actionType: string
  actionConfig: string
  errorMessage: string
}>({
  ruleCode: '',
  ruleName: '',
  ruleType: 'MATERIAL_CHECK' as RuleType,
  priority: 0,
  workstationId: '',
  conditionExpression: '',
  actionType: 'BLOCK',
  actionConfig: '',
  errorMessage: '',
})

const rules = {
  ruleCode: [{ required: true, message: t('pokayoke.ruleCodeRequired'), trigger: 'blur' }],
  ruleName: [{ required: true, message: t('pokayoke.ruleNameRequired'), trigger: 'blur' }],
  ruleType: [{ required: true, message: t('pokayoke.ruleTypeRequired'), trigger: 'change' }],
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
      ElMessage.success(t('pokayoke.updateSuccess'))
    } else {
      await ruleStore.createRule(form)
      ElMessage.success(t('pokayoke.createSuccess'))
    }
    router.push('/mes/pokayoke')
  } catch (error) {
    ElMessage.error(t('pokayoke.operationFailed'))
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