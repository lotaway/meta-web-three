<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { DISPOSITION_TYPE } from './constants'

const { t } = useI18n()

const props = defineProps<{
  visible: boolean
  loading: boolean
}>()

const emit = defineEmits<{
  'update:visible': [value: boolean]
  submit: [form: {
    dispositionType: string
    dispositionBy: string
    refundAmount: number
    replacementSkuCode: string
    replacementQuantity: number
    remark: string
  }]
}>()

const dispositionForm = ref({
  dispositionType: DISPOSITION_TYPE.REFUND,
  dispositionBy: '',
  refundAmount: 0,
  replacementSkuCode: '',
  replacementQuantity: 0,
  remark: ''
})

const dispositionTypeOptions = [
  { label: t('rma.returnTypeREFUND'), value: DISPOSITION_TYPE.REFUND },
  { label: t('rma.returnTypeREPLACEMENT'), value: DISPOSITION_TYPE.REPLACEMENT },
  { label: t('rma.returnTypeREPAIR'), value: DISPOSITION_TYPE.REPAIR },
  { label: 'SCRAP', value: DISPOSITION_TYPE.SCRAP },
  { label: 'RETURN_TO_SUPPLIER', value: DISPOSITION_TYPE.RETURN_TO_SUPPLIER }
]

watch(() => props.visible, (val) => {
  if (val) {
    dispositionForm.value = {
      dispositionType: DISPOSITION_TYPE.REFUND,
      dispositionBy: '',
      refundAmount: 0,
      replacementSkuCode: '',
      replacementQuantity: 0,
      remark: ''
    }
  }
})
</script>

<template>
  <el-dialog
    :model-value="props.visible"
    :title="t('rma.makeDisposition')"
    width="500px"
    :close-on-click-modal="false"
    @update:model-value="emit('update:visible', $event)"
  >
    <el-form :model="dispositionForm" label-width="160px">
      <el-form-item :label="t('rma.dispositionType') || 'Disposition Type'" required>
        <el-select v-model="dispositionForm.dispositionType">
          <el-option v-for="item in dispositionTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
        </el-select>
      </el-form-item>
      <el-form-item :label="t('rma.dispositionBy') || 'Disposition By'" required>
        <el-input v-model="dispositionForm.dispositionBy" placeholder="Enter disposition by" />
      </el-form-item>
      <el-form-item :label="t('rma.refundAmount') || 'Refund Amount'">
        <el-input-number v-model="dispositionForm.refundAmount" :min="0" :precision="2" />
      </el-form-item>
      <el-form-item :label="t('rma.replacementSkuCode') || 'Replacement SKU'">
        <el-input v-model="dispositionForm.replacementSkuCode" placeholder="Enter SKU code" />
      </el-form-item>
      <el-form-item :label="t('rma.replacementQuantity') || 'Replacement Qty'">
        <el-input-number v-model="dispositionForm.replacementQuantity" :min="0" />
      </el-form-item>
      <el-form-item label="Remark">
        <el-input v-model="dispositionForm.remark" type="textarea" :rows="2" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="emit('update:visible', false)">{{ t('common.cancel') }}</el-button>
      <el-button type="primary" :loading="props.loading" @click="emit('submit', dispositionForm)">{{ t('common.submit') }}</el-button>
    </template>
  </el-dialog>
</template>
