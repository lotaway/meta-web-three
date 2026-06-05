<template>
  <div class="app-container">
    <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
      <el-card class="box-card">
        <template #header><span>{{ isEdit ? t('mes.labor.edit') : t('mes.labor.create') }}</span></template>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.labor.operatorCode')" prop="operatorCode">
              <el-input v-model="form.operatorCode" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.labor.operatorName')" prop="operatorName">
              <el-input v-model="form.operatorName" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.labor.department')" prop="department">
              <el-input v-model="form.department" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.labor.shiftGroup')" prop="shiftGroup">
              <el-select v-model="form.shiftGroup" style="width: 100%">
                <el-option label="A" value="A" /><el-option label="B" value="B" />
                <el-option label="C" value="C" /><el-option label="D" value="D" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
      </el-card>
      <div style="margin-top: 20px; text-align: center;">
        <el-button @click="handleCancel">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">{{ t('common.confirm') }}</el-button>
      </div>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { createOperatorAPI, getOperatorByIdAPI } from '@/apis/labor'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()
const isEdit = computed(() => !!route.query.id)
const submitting = ref(false)
const formRef = ref()
const form = ref({ operatorCode: '', operatorName: '', department: 'Production', shiftGroup: 'A' })
const rules = {
  operatorCode: [{ required: true, message: () => t('mes.labor.operatorCodeRequired'), trigger: 'blur' }],
  operatorName: [{ required: true, message: () => t('mes.labor.operatorNameRequired'), trigger: 'blur' }],
}

async function load(id: number) {
  try {
    const res = await getOperatorByIdAPI(id)
    form.value.operatorCode = res.data.operatorCode
    form.value.operatorName = res.data.operatorName
    form.value.department = res.data.department || 'Production'
    form.value.shiftGroup = res.data.shiftGroup || 'A'
  } catch { ElMessage.error(t('mes.labor.loadFailed')) }
}
async function handleSubmit() {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return
  submitting.value = true
  try {
    await createOperatorAPI(form.value)
    ElMessage.success(t('mes.labor.createSuccess'))
    router.push({ name: 'labor' })
  } catch { ElMessage.error(t('mes.labor.submitFailed'))
  } finally { submitting.value = false }
}
function handleCancel() { router.push({ name: 'labor' }) }

onMounted(() => { if (isEdit.value && route.query.id) load(Number(route.query.id)) })
</script>
