<template>
  <div class="app-container">
    <el-form ref="formRef" :model="form" :rules="rules" label-width="120px">
      <el-card class="box-card">
        <template #header>
          <span>{{ isEdit ? t('mes.scheduling.edit') : t('mes.scheduling.create') }}</span>
        </template>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.scheduleNo')" prop="scheduleNo">
              <el-input v-model="form.scheduleNo" :disabled="isEdit" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.orderNo')" prop="orderNo">
              <el-input v-model="form.orderNo" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.productCode')" prop="productCode">
              <el-input v-model="form.productCode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.productName')" prop="productName">
              <el-input v-model="form.productName" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.quantity')" prop="quantity">
              <el-input-number v-model="form.quantity" :min="1" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.priority')" prop="priority">
              <el-select v-model="form.priority" style="width: 100%">
                <el-option label="LOW" value="LOW" />
                <el-option label="NORMAL" value="NORMAL" />
                <el-option label="HIGH" value="HIGH" />
                <el-option label="URGENT" value="URGENT" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.dueDate')" prop="dueDate">
              <el-date-picker v-model="form.dueDate" type="datetime" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('mes.scheduling.workshop')" prop="workshopId">
              <el-input v-model="form.workshopId" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('mes.scheduling.routeCode')" prop="routeCode">
          <el-input v-model="form.routeCode" />
        </el-form-item>
      </el-card>
      <div style="margin-top: 20px; text-align: center;">
        <el-button @click="handleCancel">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">
          {{ t('common.confirm') }}
        </el-button>
      </div>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { createScheduleOrderAPI, getScheduleOrderByIdAPI, type ScheduleOrder } from '@/apis/scheduling'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const isEdit = computed(() => !!route.query.id)
const submitting = ref(false)
const formRef = ref()

const form = ref({
  scheduleNo: '',
  orderNo: '',
  productCode: '',
  productName: '',
  quantity: 1,
  priority: 'NORMAL',
  dueDate: null as string | null,
  workshopId: 'Workshop-001',
  routeCode: '',
})

const rules = {
  scheduleNo: [{ required: true, message: () => t('mes.scheduling.scheduleNoRequired'), trigger: 'blur' }],
  orderNo: [{ required: true, message: () => t('mes.scheduling.orderNoRequired'), trigger: 'blur' }],
  productCode: [{ required: true, message: () => t('mes.scheduling.productCodeRequired'), trigger: 'blur' }],
  productName: [{ required: true, message: () => t('mes.scheduling.productNameRequired'), trigger: 'blur' }],
  quantity: [{ required: true, message: () => t('mes.scheduling.quantityRequired'), trigger: 'blur' }],
  priority: [{ required: true, message: () => t('mes.scheduling.priorityRequired'), trigger: 'change' }],
  workshopId: [{ required: true, message: () => t('mes.scheduling.workshopIdRequired'), trigger: 'blur' }],
}

async function loadOrder(id: number) {
  try {
    const res = await getScheduleOrderByIdAPI(id)
    const order: ScheduleOrder = res.data
    form.value.scheduleNo = order.scheduleNo
    form.value.orderNo = order.orderNo
    form.value.productCode = order.productCode
    form.value.productName = order.productName
    form.value.quantity = order.quantity
    form.value.priority = order.priority
    form.value.dueDate = order.dueDate || null
    form.value.workshopId = order.workshopId
    form.value.routeCode = order.routeCode || ''
  } catch {
    ElMessage.error(t('mes.scheduling.loadFailed'))
  }
}

async function handleSubmit() {
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return

  submitting.value = true
  try {
    await createScheduleOrderAPI({
      scheduleNo: form.value.scheduleNo,
      orderNo: form.value.orderNo,
      productCode: form.value.productCode,
      productName: form.value.productName,
      quantity: form.value.quantity,
      priority: form.value.priority as any,
      dueDate: form.value.dueDate || undefined,
      workshopId: form.value.workshopId,
      routeCode: form.value.routeCode || undefined,
    })
    ElMessage.success(t('mes.scheduling.createSuccess'))
    router.push({ name: 'scheduling' })
  } catch {
    ElMessage.error(t('mes.scheduling.submitFailed'))
  } finally {
    submitting.value = false
  }
}

function handleCancel() {
  router.push({ name: 'scheduling' })
}

onMounted(() => {
  if (isEdit.value && route.query.id) {
    loadOrder(Number(route.query.id))
  }
})
</script>
