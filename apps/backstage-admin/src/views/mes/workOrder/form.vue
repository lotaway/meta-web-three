<template>
  <div class="app-container">
    <el-card class="form-card">
      <template #header>
        <div class="card-header">
          <span>{{ isEdit ? t('mes.workOrder.edit') : t('mes.workOrder.add') }}</span>
        </div>
      </template>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-width="140px"
        class="work-order-form"
      >
        <el-form-item :label="t('mes.workOrder.workOrderNo')" prop="workOrderNo">
          <el-input
            v-model="form.workOrderNo"
            :placeholder="t('mes.workOrder.workOrderNoPlaceholder')"
            :disabled="isEdit"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.productCode')" prop="productCode">
          <el-input
            v-model="form.productCode"
            :placeholder="t('mes.workOrder.productCodePlaceholder')"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.productName')" prop="productName">
          <el-input
            v-model="form.productName"
            :placeholder="t('mes.workOrder.productNamePlaceholder')"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.quantity')" prop="quantity">
          <el-input-number
            v-model="form.quantity"
            :min="1"
            :placeholder="t('mes.workOrder.quantityRequired')"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.typeCode')">
          <el-select
            v-model="form.typeCode"
            :placeholder="t('mes.workOrder.typeCodePlaceholder')"
            clearable
          >
            <el-option value="NORMAL" :label="t('mes.workOrder.typeCodeNORMAL')" />
            <el-option value="REWORK" :label="t('mes.workOrder.typeCodeREWORK')" />
            <el-option value="REPAIR" :label="t('mes.workOrder.typeCodeREPAIR')" />
            <el-option value="SAMPLE" :label="t('mes.workOrder.typeCodeSAMPLE')" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.priority')">
          <el-select
            v-model="form.priority"
            :placeholder="t('mes.workOrder.priorityPlaceholder')"
            clearable
          >
            <el-option value="LOW" :label="t('mes.workOrder.priorityLOW')" />
            <el-option value="NORMAL" :label="t('mes.workOrder.priorityNORMAL')" />
            <el-option value="HIGH" :label="t('mes.workOrder.priorityHIGH')" />
            <el-option value="URGENT" :label="t('mes.workOrder.priorityURGENT')" />
          </el-select>
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.workshopId')">
          <el-input
            v-model="form.workshopId"
            :placeholder="t('mes.workOrder.workshopIdPlaceholder')"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.processRouteId')">
          <el-input
            v-model="form.processRouteId"
            :placeholder="t('mes.workOrder.processRouteIdPlaceholder')"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.plannedStartTime')">
          <el-date-picker
            v-model="form.plannedStartTime"
            type="datetime"
            :placeholder="t('mes.workOrder.plannedStartTime')"
            value-format="YYYY-MM-DDTHH:mm:ss"
          />
        </el-form-item>

        <el-form-item :label="t('mes.workOrder.plannedEndTime')">
          <el-date-picker
            v-model="form.plannedEndTime"
            type="datetime"
            :placeholder="t('mes.workOrder.plannedEndTime')"
            value-format="YYYY-MM-DDTHH:mm:ss"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="handleSubmit">
            {{ t('mes.workOrder.save') }}
          </el-button>
          <el-button @click="handleBack">
            {{ t('mes.workOrder.cancel') }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import type { FormInstance, FormRules } from 'element-plus'
import {
  createWorkOrderAPI,
  updateWorkOrderAPI,
  getWorkOrderByIdAPI,
  type WorkOrder,
  type CreateWorkOrderRequest,
  type UpdateWorkOrderRequest,
  type WorkOrderType,
  type WorkOrderPriority
} from '@/apis/workOrder'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const formRef = ref<FormInstance>()
const isEdit = computed(() => !!route.query.id)
const isLoading = ref(false)

const form = ref<CreateWorkOrderRequest & UpdateWorkOrderRequest>({
  workOrderNo: '',
  productCode: '',
  productName: '',
  quantity: 1,
  workshopId: '',
  processRouteId: '',
  typeCode: undefined,
  priority: undefined,
  plannedStartTime: undefined,
  plannedEndTime: undefined,
})

const rules = ref<FormRules>({
  workOrderNo: [
    { required: true, message: t('mes.workOrder.workOrderNoRequired'), trigger: 'blur' },
  ],
  productCode: [
    { required: true, message: t('mes.workOrder.productCodeRequired'), trigger: 'blur' },
  ],
  productName: [
    { required: true, message: t('mes.workOrder.productNameRequired'), trigger: 'blur' },
  ],
  quantity: [
    { required: true, message: t('mes.workOrder.quantityRequired'), trigger: 'blur' },
  ],
})

const handleSubmit = async () => {
  if (!formRef.value) return

  await formRef.value.validate(async (valid) => {
    if (!valid) return

    isLoading.value = true
    try {
      if (isEdit.value) {
        const id = Number(route.query.id)
        await updateWorkOrderAPI(id, {
          productCode: form.value.productCode,
          productName: form.value.productName,
          quantity: form.value.quantity,
          workshopId: form.value.workshopId,
          processRouteId: form.value.processRouteId,
          priority: form.value.priority as WorkOrderPriority,
          plannedStartTime: form.value.plannedStartTime,
          plannedEndTime: form.value.plannedEndTime,
        })
        ElMessage.success(t('mes.workOrder.updateSuccess'))
      } else {
        await createWorkOrderAPI({
          workOrderNo: form.value.workOrderNo,
          productCode: form.value.productCode,
          productName: form.value.productName,
          quantity: form.value.quantity,
          workshopId: form.value.workshopId,
          processRouteId: form.value.processRouteId,
          typeCode: form.value.typeCode as WorkOrderType,
        })
        ElMessage.success(t('mes.workOrder.createSuccess'))
      }
      router.push({ name: 'workOrder' })
    } catch (error) {
      ElMessage.error(t('mes.workOrder.submitFailed'))
    } finally {
      isLoading.value = false
    }
  })
}

const handleBack = () => {
  router.push({ name: 'workOrder' })
}

const loadData = async () => {
  if (!isEdit.value) return

  isLoading.value = true
  try {
    const id = Number(route.query.id)
    const response = await getWorkOrderByIdAPI(id)
    if (response.data) {
      const data = response.data
      form.value = {
        workOrderNo: data.workOrderNo || '',
        productCode: data.productCode || '',
        productName: data.productName || '',
        quantity: data.quantity || 1,
        workshopId: data.workshopId || '',
        processRouteId: data.processRouteId || '',
        typeCode: data.typeCode as WorkOrderType | undefined,
        priority: data.priority as WorkOrderPriority | undefined,
        plannedStartTime: data.plannedStartTime,
        plannedEndTime: data.plannedEndTime,
      }
    }
  } catch (error) {
    ElMessage.error(t('mes.workOrder.loadFailed'))
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.form-card {
  max-width: 800px;
  margin: 0 auto;
}
.card-header {
  font-size: 18px;
  font-weight: bold;
}
.work-order-form {
  padding: 20px;
}
</style>