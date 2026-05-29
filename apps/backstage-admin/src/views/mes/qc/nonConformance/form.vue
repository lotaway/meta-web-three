<template>
  <div class="non-conformance-form-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.qc.nonConformance.edit') : t('mes.qc.nonConformance.add') }}</span>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="150px">
        <el-form-item :label="t('mes.qc.nonConformance.dispositionCode')" prop="dispositionCode">
          <el-input v-model="form.dispositionCode" :placeholder="t('mes.qc.nonConformance.dispositionCodePlaceholder')" :disabled="isEdit" />
        </el-form-item>
        <el-form-item :label="t('mes.qc.nonConformance.dispositionName')" prop="dispositionName">
          <el-input v-model="form.dispositionName" :placeholder="t('mes.qc.nonConformance.dispositionNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.qc.nonConformance.type')" prop="type">
          <el-select v-model="form.type" :placeholder="t('mes.qc.nonConformance.typePlaceholder')">
            <el-option :label="t('mes.qc.nonConformance.typeScrap')" value="SCRAP" />
            <el-option :label="t('mes.qc.nonConformance.typeRework')" value="REWORK" />
            <el-option :label="t('mes.qc.nonConformance.typeReturn')" value="RETURN" />
            <el-option :label="t('mes.qc.nonConformance.typeUseAsIs')" value="USE_AS_IS" />
            <el-option :label="t('mes.qc.nonConformance.type降级使用')" value="降级使用" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.qc.nonConformance.sortOrder')" prop="sortOrder">
          <el-input-number v-model="form.sortOrder" :min="0" :max="9999" />
        </el-form-item>

        <el-divider>{{ t('mes.qc.nonConformance.steps') }}</el-divider>

        <div v-for="(step, index) in form.steps" :key="index" class="step-item">
          <el-card>
            <el-row :gutter="20">
              <el-col :span="5">
                <el-form-item :label="t('mes.qc.nonConformance.stepOrder')" label-width="100px">
                  <el-input-number v-model="step.stepOrder" :min="1" :max="20" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item :label="t('mes.qc.nonConformance.stepName')" label-width="100px">
                  <el-input v-model="step.stepName" :placeholder="t('mes.qc.nonConformance.stepNamePlaceholder')" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item :label="t('mes.qc.nonConformance.action')" label-width="100px">
                  <el-select v-model="step.action" :placeholder="t('mes.qc.nonConformance.actionPlaceholder')">
                    <el-option :label="t('mes.qc.nonConformance.actionApprove')" value="APPROVE" />
                    <el-option :label="t('mes.qc.nonConformance.actionNotify')" value="NOTIFY" />
                    <el-option :label="t('mes.qc.nonConformance.actionExecute')" value="EXECUTE" />
                    <el-option :label="t('mes.qc.nonConformance.actionRecord')" value="RECORD" />
                  </el-select>
                </el-form-item>
              </el-col>
              <el-col :span="5">
                <el-form-item :label="t('mes.qc.nonConformance.assigneeRole')" label-width="120px">
                  <el-input v-model="step.assigneeRole" :placeholder="t('mes.qc.nonConformance.assigneeRolePlaceholder')" />
                </el-form-item>
              </el-col>
              <el-col :span="2">
                <el-button type="danger" link @click="removeStep(index)">{{ t('common.delete') }}</el-button>
              </el-col>
            </el-row>
            <el-row :gutter="20">
              <el-col :span="5">
                <el-form-item :label="t('mes.qc.nonConformance.requiresApproval')" label-width="100px">
                  <el-switch v-model="step.requiresApproval" />
                </el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item :label="t('mes.qc.nonConformance.timeoutHours')" label-width="100px">
                  <el-input-number v-model="step.timeoutHours" :min="0" :max="168" />
                </el-form-item>
              </el-col>
            </el-row>
          </el-card>
        </div>

        <el-form-item>
          <el-button type="primary" @click="submitForm" :loading="submitting">{{ t('mes.qc.nonConformance.save') }}</el-button>
          <el-button @click="goBack">{{ t('mes.qc.nonConformance.cancel') }}</el-button>
          <el-button type="success" @click="addStep">{{ t('mes.qc.nonConformance.addStep') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, FormInstance, FormRules } from 'element-plus'
import type { NonConformanceDisposition, DispositionStep } from '@/apis/qc'
import {
  getNonConformanceDispositionByIdAPI,
  createNonConformanceDispositionAPI,
  updateNonConformanceDispositionAPI
} from '@/apis/qc'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const form = reactive<{
  dispositionCode: string
  dispositionName: string
  type: string
  sortOrder: number
  steps: DispositionStep[]
}>({
  dispositionCode: '',
  dispositionName: '',
  type: '',
  sortOrder: 0,
  steps: []
})

const rules = reactive<FormRules>({
  dispositionCode: [
    { required: true, message: t('mes.qc.nonConformance.dispositionCodePlaceholder'), trigger: 'blur' }
  ],
  dispositionName: [
    { required: true, message: t('mes.qc.nonConformance.dispositionNamePlaceholder'), trigger: 'blur' }
  ],
  type: [
    { required: true, message: t('mes.qc.nonConformance.typePlaceholder'), trigger: 'change' }
  ]
})

const loadData = async () => {
  if (route.query.id) {
    try {
      const res = await getNonConformanceDispositionByIdAPI(Number(route.query.id))
      if (res.data) {
        form.dispositionCode = res.data.dispositionCode
        form.dispositionName = res.data.dispositionName
        form.type = res.data.type
        form.sortOrder = res.data.sortOrder || 0
        form.steps = res.data.steps || []
      }
    } catch (error) {
      ElMessage.error(t('message.loadFailed'))
    }
  }
}

const addStep = () => {
  form.steps.push({
    stepOrder: form.steps.length + 1,
    stepName: '',
    action: '',
    assigneeRole: '',
    requiresApproval: false,
    timeoutHours: 24
  })
}

const removeStep = (index: number) => {
  form.steps.splice(index, 1)
}

const submitForm = async () => {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    submitting.value = true
    try {
      const data: Partial<NonConformanceDisposition> = {
        dispositionCode: form.dispositionCode,
        dispositionName: form.dispositionName,
        type: form.type as any,
        sortOrder: form.sortOrder,
        steps: form.steps
      }

      if (isEdit.value) {
        await updateNonConformanceDispositionAPI(Number(route.query.id), data)
      } else {
        await createNonConformanceDispositionAPI(data)
      }

      ElMessage.success(t('message.saveSuccess'))
      goBack()
    } catch (error) {
      ElMessage.error(t('message.saveFailed'))
    } finally {
      submitting.value = false
    }
  })
}

const goBack = () => {
  router.push({ path: '/mes/nonConformance' })
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.non-conformance-form-container {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: bold;
}

.step-item {
  margin-bottom: 15px;
}
</style>