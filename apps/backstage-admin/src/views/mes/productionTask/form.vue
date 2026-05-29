<template>
  <div class="app-container">
    <el-page-header :title="isEdit ? t('mes.productionTask.edit') : t('mes.productionTask.add')" @back="goBack">
      <template #content>
        <span class="text-lg">{{ isEdit ? t('mes.productionTask.edit') : t('mes.productionTask.add') }}</span>
      </template>
    </el-page-header>

    <el-card v-loading="loading" class="mt-4">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="140px">
        <el-form-item :label="t('mes.productionTask.taskNo')" prop="taskNo">
          <el-input v-model="form.taskNo" :placeholder="t('mes.productionTask.taskNoPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.productionTask.workOrderId')" prop="workOrderId">
          <el-input-number v-model="form.workOrderId" :min="1" />
        </el-form-item>
        <el-form-item :label="t('mes.productionTask.processCode')" prop="processCode">
          <el-input v-model="form.processCode" :placeholder="t('mes.productionTask.processCodePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.productionTask.workstationId')" prop="workstationId">
          <el-input v-model="form.workstationId" :placeholder="t('mes.productionTask.workstationIdPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('mes.productionTask.quantity')" prop="quantity">
          <el-input-number v-model="form.quantity" :min="1" />
        </el-form-item>
        <el-form-item :label="t('mes.productionTask.operatorId')" prop="operatorId">
          <el-input v-model="form.operatorId" :placeholder="t('mes.productionTask.operatorIdPlaceholder')" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSubmit">{{ t('common.save') }}</el-button>
          <el-button @click="goBack">{{ t('common.cancel') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { ref, onMounted, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { 
  getTaskByIdAPI, 
  createTaskAPI, 
  updateTaskAPI,
  type CreateTaskRequest 
} from '@/apis/productionTask'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const loading = ref(false)
const isEdit = computed(() => !!route.query.id)

const form = ref<CreateTaskRequest>({
  taskNo: '',
  workOrderId: 0,
  processCode: '',
  workstationId: '',
  quantity: 1,
  operatorId: '',
})

const rules: FormRules = {
  taskNo: [
    { required: true, message: t('mes.productionTask.taskNoRequired'), trigger: 'blur' },
  ],
  workOrderId: [
    { required: true, message: t('mes.productionTask.workOrderIdRequired'), trigger: 'blur' },
  ],
  quantity: [
    { required: true, message: t('mes.productionTask.quantityRequired'), trigger: 'blur' },
  ],
}

const goBack = () => {
  router.back()
}

const loadTask = async () => {
  if (!isEdit.value) return
  loading.value = true
  try {
    const id = Number(route.query.id)
    const response = await getTaskByIdAPI(id)
    const task = response.data
    if (task) {
      form.value = {
        taskNo: task.taskNo,
        workOrderId: task.workOrderId,
        processCode: task.processCode || '',
        workstationId: task.workstationId || '',
        quantity: task.quantity,
        operatorId: task.operatorId || '',
      }
    }
  } catch (error) {
    ElMessage.error(t('mes.productionTask.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      if (isEdit.value) {
        const id = Number(route.query.id)
        await updateTaskAPI(id, form.value)
        ElMessage.success(t('mes.productionTask.updateSuccess'))
      } else {
        await createTaskAPI(form.value)
        ElMessage.success(t('mes.productionTask.createSuccess'))
      }
      router.back()
    } catch (error: unknown) {
      const err = error as { response?: { data?: { message?: string } } }
      ElMessage.error(err.response?.data?.message || t('mes.productionTask.submitFailed'))
    } finally {
      loading.value = false
    }
  })
}

onMounted(() => {
  loadTask()
})
</script>

<style scoped>
.mt-4 {
  margin-top: 16px;
}
.text-lg {
  font-size: 18px;
}
</style>
