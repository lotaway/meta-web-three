<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getOrderSettingByIdAPI, orderSettingUpdateByIdAPI } from '@/apis/orderSetting'
import type { FormInstance, FormRules } from 'element-plus'
import type { OmsOrderSetting } from '@/types/orderSetting'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Default order settings data
const defaultOrderSetting = {
  id: 1,
  flashOrderOvertime: 0,
  normalOrderOvertime: 0,
  confirmOvertime: 0,
  finishOvertime: 0,
  commentOvertime: 0
}
// Order settings data
const orderSetting = ref<OmsOrderSetting>(Object.assign({}, defaultOrderSetting))
// Get details
const getDetail = async () => {
  const response = await getOrderSettingByIdAPI(orderSetting.value.id)
  orderSetting.value = response.data
}

// Get details after mount
onMounted(() => {
  getDetail()
})

// Order settings form ref
const orderSettingForm = ref<FormInstance>()
// Time validation rules
const checkTime = (rule: unknown, value: string, callback: (error?: Error) => void) => {
  if (!value) {
    return callback(new Error(t('oms.orderSetting.timeRequired')))
  }
  const intValue = parseInt(value)
  if (!Number.isInteger(intValue)) {
    return callback(new Error(t('oms.orderSetting.numberRequired')))
  }
  callback()
}
// Form validation rules
const rules = ref<FormRules>({
  flashOrderOvertime: { validator: checkTime, trigger: 'blur' },
  normalOrderOvertime: { validator: checkTime, trigger: 'blur' },
  confirmOvertime: { validator: checkTime, trigger: 'blur' },
  finishOvertime: { validator: checkTime, trigger: 'blur' },
  commentOvertime: { validator: checkTime, trigger: 'blur' }
})

const submitOrderSetting = async () => {
  await ElMessageBox.confirm(t('oms.orderSetting.confirmSubmit'), t('oms.orderSetting.confirmTitle'), {
    confirmButtonText: t('common.confirmText'),
    cancelButtonText: t('common.cancelText'),
    type: 'warning'
  })
  await orderSettingUpdateByIdAPI(1, orderSetting.value)
  ElMessage({
    type: 'success',
    message: t('oms.orderSetting.submitSuccess'),
    duration: 1000
  })
}

// Confirm and submit form
const confirm = async () => {
  if (!orderSettingForm.value) return
  const valid = await orderSettingForm.value.validate()
  if (!valid) {
    ElMessage({
      message: t('oms.orderSetting.submitFailed'),
      type: 'warning'
    })
    return false
  }
  await submitOrderSetting()
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-form :model="orderSetting" ref="orderSettingForm" :rules="rules" label-width="150px">
      <el-form-item :label="t('oms.orderSetting.flashOrderOvertime') + '：'" prop="flashOrderOvertime">
        <el-input v-model="orderSetting.flashOrderOvertime" class="input-width">
          <template #append>{{ t('oms.orderSetting.minutes') }}</template>
        </el-input>
        <span class="note-margin">{{ t('oms.orderSetting.flashOrderTip') }}</span>
      </el-form-item>
      <el-form-item :label="t('oms.orderSetting.normalOrderOvertime') + '：'" prop="normalOrderOvertime">
        <el-input v-model="orderSetting.normalOrderOvertime" class="input-width">
          <template #append>{{ t('oms.orderSetting.minutes') }}</template>
        </el-input>
        <span class="note-margin">{{ t('oms.orderSetting.normalOrderTip') }}</span>
      </el-form-item>
      <el-form-item :label="t('oms.orderSetting.confirmOvertime') + '：'" prop="confirmOvertime">
        <el-input v-model="orderSetting.confirmOvertime" class="input-width">
          <template #append>{{ t('oms.orderSetting.days') }}</template>
        </el-input>
        <span class="note-margin">{{ t('oms.orderSetting.confirmOrderTip') }}</span>
      </el-form-item>
      <el-form-item :label="t('oms.orderSetting.finishOrderOvertime') + '：'" prop="finishOvertime">
        <el-input v-model="orderSetting.finishOvertime" class="input-width">
          <template #append>{{ t('oms.orderSetting.days') }}</template>
        </el-input>
        <span class="note-margin">{{ t('oms.orderSetting.finishOrderTip') }}</span>
      </el-form-item>
      <el-form-item :label="t('oms.orderSetting.commentOrderOvertime') + '：'" prop="commentOvertime">
        <el-input v-model="orderSetting.commentOvertime" class="input-width">
          <template #append>{{ t('oms.orderSetting.days') }}</template>
        </el-input>
        <span class="note-margin">{{ t('oms.orderSetting.commentOrderTip') }}</span>
      </el-form-item>
      <el-form-item>
        <el-button @click="confirm()" type="primary">{{ t('oms.orderSetting.submit') }}</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped>
.input-width {
  width: 50%
}

.note-margin {
  margin-left: 15px
}
</style>
