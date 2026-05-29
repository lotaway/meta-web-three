<template>
  <div class="process-template-form-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.processTemplate.edit') : t('mes.processTemplate.add') }}</span>
          <el-button @click="handleBack">{{ t('mes.processTemplate.back') }}</el-button>
        </div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-width="150px">
        <el-form-item :label="t('mes.processTemplate.templateCode')" prop="templateCode">
          <el-input v-model="form.templateCode" :placeholder="t('mes.processTemplate.templateCodePlaceholder')" :disabled="isEdit" />
        </el-form-item>
        <el-form-item :label="t('mes.processTemplate.templateName')" prop="templateName">
          <el-input v-model="form.templateName" :placeholder="t('mes.processTemplate.templateNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.processTemplate.description')" prop="description">
          <el-input v-model="form.description" type="textarea" :rows="3" :placeholder="t('mes.processTemplate.descriptionPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.processTemplate.flowData')" prop="flowData">
          <el-input v-model="form.flowData" type="textarea" :rows="10" :placeholder="t('mes.processTemplate.flowDataPlaceholder')" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSubmit" :loading="submitting">
            {{ t('mes.processTemplate.submit') }}
          </el-button>
          <el-button @click="handleBack">{{ t('mes.processTemplate.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import {
  getProcessTemplateAPI,
  createProcessTemplateAPI,
  updateProcessTemplateAPI,
} from '@/apis/processFlow'
import type { ProcessFlowTemplate } from '@/apis/processFlow'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const submitting = ref(false)
const isEdit = computed(() => !!route.query.id)

const form = reactive<ProcessFlowTemplate>({
  templateCode: '',
  templateName: '',
  description: '',
  flowData: '',
  status: 'DRAFT',
})

const rules = reactive<FormRules>({
  templateCode: [
    { required: true, message: t('mes.processTemplate.templateCodeRequired'), trigger: 'blur' },
  ],
  templateName: [
    { required: true, message: t('mes.processTemplate.templateNameRequired'), trigger: 'blur' },
  ],
})

const handleSubmit = async () => {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    submitting.value = true
    try {
      if (isEdit.value) {
        await updateProcessTemplateAPI(Number(route.query.id), form)
        ElMessage.success(t('mes.processTemplate.updateSuccess'))
      } else {
        await createProcessTemplateAPI(form)
        ElMessage.success(t('mes.processTemplate.createSuccess'))
      }
      router.push('/mes/process-template')
    } catch (error) {
      ElMessage.error(t('mes.processTemplate.submitFailed') + (error as Error).message)
    } finally {
      submitting.value = false
    }
  })
}

const handleBack = () => {
  router.back()
}

const loadData = async () => {
  if (!isEdit.value) return

  try {
    const res = await getProcessTemplateAPI(Number(route.query.id))
    if (res.data) {
      Object.assign(form, res.data)
    }
  } catch (error) {
    ElMessage.error(t('mes.processTemplate.loadFailed'))
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.process-template-form-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>