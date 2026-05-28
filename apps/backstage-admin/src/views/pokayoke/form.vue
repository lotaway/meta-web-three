<template>
  <div class="pokayoke-rule-form-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.pokayoke.edit') : t('mes.pokayoke.add') }}</span>
          <el-button @click="handleBack">{{ t('mes.pokayoke.back') }}</el-button>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.pokayoke.ruleCode')" prop="ruleCode">
              <el-input v-model="form.ruleCode" :placeholder="t('mes.pokayoke.ruleCodePlaceholder')" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.pokayoke.ruleName')" prop="ruleName">
              <el-input v-model="form.ruleName" :placeholder="t('mes.pokayoke.ruleNamePlaceholder')" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.pokayoke.ruleType')" prop="ruleType">
              <el-select v-model="form.ruleType" :placeholder="t('mes.pokayoke.ruleTypePlaceholder')">
                <el-option :label="t('mes.pokayoke.typeMaterialCheck')" value="MATERIAL_CHECK" />
                <el-option :label="t('mes.pokayoke.typeSequenceCheck')" value="SEQUENCE_CHECK" />
                <el-option :label="t('mes.pokayoke.typeParameterCheck')" value="PARAMETER_CHECK" />
                <el-option :label="t('mes.pokayoke.typeStationCheck')" value="STATION_CHECK" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.pokayoke.priority')">
              <el-input-number v-model="form.priority" :min="0" :max="100" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.pokayoke.workstation')">
              <el-input v-model="form.workstationId" :placeholder="t('mes.pokayoke.workstationIdPlaceholder')" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.pokayoke.errorMessage')">
              <el-input v-model="form.errorMessage" :placeholder="t('mes.pokayoke.errorMessagePlaceholder')" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item :label="t('mes.pokayoke.conditionExpression')">
          <el-input 
            v-model="form.conditionExpression" 
            type="textarea" 
            :rows="4"
            :placeholder="t('mes.pokayoke.conditionExpressionPlaceholder')" 
          />
          <div class="form-tip">
            {{ t('mes.pokayoke.conditionExpressionTip') }}
          </div>
        </el-form-item>

        <el-form-item :label="t('mes.pokayoke.actionType')">
          <el-select v-model="form.actionType" :placeholder="t('mes.pokayoke.actionTypePlaceholder')">
            <el-option :label="t('mes.pokayoke.actionBlock')" value="BLOCK" />
            <el-option :label="t('mes.pokayoke.actionWarning')" value="WARNING" />
            <el-option :label="t('mes.pokayoke.actionLog')" value="LOG" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.pokayoke.actionConfig')">
          <el-input 
            v-model="form.actionConfig" 
            type="textarea" 
            :rows="3"
            :placeholder="t('mes.pokayoke.actionConfigPlaceholder')" 
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">{{ t('mes.pokayoke.save') }}</el-button>
          <el-button @click="handleBack">{{ t('mes.pokayoke.cancel') }}</el-button>
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

const { t } = useI18n()
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
  ruleCode: [{ required: true, message: t('mes.pokayoke.ruleCodeRequired'), trigger: 'blur' }],
  ruleName: [{ required: true, message: t('mes.pokayoke.ruleNameRequired'), trigger: 'blur' }],
  ruleType: [{ required: true, message: t('mes.pokayoke.ruleTypeRequired'), trigger: 'change' }],
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
      ElMessage.success(t('mes.pokayoke.updateSuccess'))
    } else {
      await ruleStore.createRule(form)
      ElMessage.success(t('mes.pokayoke.createSuccess'))
    }
    router.push('/mes/pokayoke')
  } catch (error) {
    ElMessage.error(t('mes.pokayoke.operationFailed'))
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