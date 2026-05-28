<script setup lang="ts">
import { ref, computed } from 'vue'
import { t } from '@/locales'

const i18n = (key: string) => t(`oms.order.${key}`)

const props = defineProps({
  modelValue: Boolean
})

const emit = defineEmits(['update:modelValue'])

const defaultLogisticsList = [
  { name: i18n('logisticsSubmitted'), time: '' },
  { name: i18n('logisticsPaid'), time: '' },
  { name: i18n('logisticsScanning'), time: '' },
  { name: i18n('logisticsSorting'), time: '' },
  { name: i18n('logisticsDispatched'), time: '' },
  { name: i18n('logisticsArrived'), time: '' },
  { name: i18n('logisticsSigned'), time: '' }
]

const logisticsList = ref(Object.assign([], defaultLogisticsList))

const visible = computed({
  get() {
    return props.modelValue
  },
  set(visible) {
    emit('update:modelValue', visible)
  }
})

const emitInput = (val: boolean) => {
  emit('update:modelValue', val)
}

const handleClose = () => {
  emitInput(false)
}
</script>

<template>
  <el-dialog :title="i18n('logisticsTitle')" v-model="visible" :before-close="handleClose" width="40%">
    <el-steps direction="vertical" :active="6" finish-status="success" space="50px">
      <el-step v-for="item in logisticsList" :key="item.name" :title="item.name" :description="item.time"></el-step>
    </el-steps>
  </el-dialog>
</template>

<style></style>