<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { INSPECTION_RESULT } from './constants'

const { t } = useI18n()

const props = defineProps<{
  visible: boolean
  loading: boolean
}>()

const emit = defineEmits<{
  'update:visible': [value: boolean]
  submit: [form: {
    inspector: string
    result: string
    conclusion: string
    totalInspected: number
    totalPassed: number
    totalFailed: number
    remark: string
  }]
}>()

const INSPECTION_CONCLUSION = {
  NO_ISSUE: 'NO_ISSUE',
  MINOR_DEFECT: 'MINOR_DEFECT',
  MAJOR_DEFECT: 'MAJOR_DEFECT'
}

const inspectionForm = ref({
  inspector: '',
  result: INSPECTION_RESULT.PASS,
  conclusion: INSPECTION_CONCLUSION.NO_ISSUE,
  totalInspected: 0,
  totalPassed: 0,
  totalFailed: 0,
  remark: ''
})

const inspectionResultOptions = [
  { label: 'PASS', value: INSPECTION_RESULT.PASS },
  { label: 'FAIL', value: INSPECTION_RESULT.FAIL },
  { label: 'PARTIAL', value: INSPECTION_RESULT.PARTIAL }
]

const inspectionConclusionOptions = [
  { label: t('rma.conclusionNoIssue'), value: INSPECTION_CONCLUSION.NO_ISSUE },
  { label: t('rma.conclusionMinorDefect'), value: INSPECTION_CONCLUSION.MINOR_DEFECT },
  { label: t('rma.conclusionMajorDefect'), value: INSPECTION_CONCLUSION.MAJOR_DEFECT }
]

watch(() => props.visible, (val) => {
  if (val) {
    inspectionForm.value = {
      inspector: '',
      result: INSPECTION_RESULT.PASS,
      conclusion: INSPECTION_CONCLUSION.NO_ISSUE,
      totalInspected: 0,
      totalPassed: 0,
      totalFailed: 0,
      remark: ''
    }
  }
})
</script>

<template>
  <el-dialog
    :model-value="props.visible"
    :title="t('rma.recordInspection')"
    width="500px"
    :close-on-click-modal="false"
    @update:model-value="emit('update:visible', $event)"
  >
    <el-form :model="inspectionForm" label-width="120px">
      <el-form-item :label="t('rma.inspector') || 'Inspector'" required>
        <el-input v-model="inspectionForm.inspector" placeholder="Enter inspector" />
      </el-form-item>
      <el-form-item :label="t('rma.result') || 'Result'" required>
        <el-select v-model="inspectionForm.result">
          <el-option v-for="item in inspectionResultOptions" :key="item.value" :label="item.label" :value="item.value" />
        </el-select>
      </el-form-item>
      <el-form-item :label="t('rma.totalInspected') || 'Total Inspected'" required>
        <el-input-number v-model="inspectionForm.totalInspected" :min="0" />
      </el-form-item>
      <el-form-item :label="t('rma.totalPassed') || 'Total Passed'" required>
        <el-input-number v-model="inspectionForm.totalPassed" :min="0" />
      </el-form-item>
      <el-form-item :label="t('rma.totalFailed') || 'Total Failed'" required>
        <el-input-number v-model="inspectionForm.totalFailed" :min="0" />
      </el-form-item>
      <el-form-item :label="t('rma.conclusion') || 'Conclusion'">
        <el-select v-model="inspectionForm.conclusion">
          <el-option v-for="item in inspectionConclusionOptions" :key="item.value" :label="item.label" :value="item.value" />
        </el-select>
      </el-form-item>
      <el-form-item label="Remark">
        <el-input v-model="inspectionForm.remark" type="textarea" :rows="2" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="emit('update:visible', false)">{{ t('common.cancel') }}</el-button>
      <el-button type="primary" :loading="props.loading" @click="emit('submit', inspectionForm)">{{ t('common.submit') }}</el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
@media (max-width: 768px) {
  :deep(.el-dialog) {
    width: 92% !important;
    max-width: 92% !important;
  }
}
</style>
