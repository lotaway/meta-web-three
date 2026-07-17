<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import type { RmaOrder } from '@/apis/rma'
import { formatDateTime } from '@/utils/datetime'
import { STATUS_TAG_TYPE_MAP } from './constants'

const { t } = useI18n()

const props = defineProps<{
  visible: boolean
  detailData: RmaOrder | null
}>()

const emit = defineEmits<{
  'update:visible': [value: boolean]
}>()

const getStatusType = (status: string): string => {
  return STATUS_TAG_TYPE_MAP[status] || 'info'
}

const getStatusLabel = (status: string) => {
  return t(`rma.status${status}`)
}

const formatAmount = (amount?: number) => {
  if (amount === null || amount === undefined) return '-'
  return `$${amount.toFixed(2)}`
}
</script>

<template>
  <el-dialog
    :model-value="props.visible"
    :title="t('rma.viewRma')"
    width="800px"
    :close-on-click-modal="false"
    @update:model-value="emit('update:visible', $event)"
  >
    <template v-if="detailData">
      <el-descriptions :column="2" border>
        <el-descriptions-item :label="t('rma.rmaNo')">{{ detailData.rmaNo }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.orderNo')">{{ detailData.orderNo }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.returnType')">{{ t(`rma.returnType${detailData.returnType}`) || detailData.returnType }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.customerName')">{{ detailData.customerName }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.contactPhone')">{{ detailData.contactPhone || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.status')">
          <el-tag :type="getStatusType(detailData.status)">{{ getStatusLabel(detailData.status) }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item :label="t('rma.reasonCode')">{{ detailData.reasonCode || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.reasonDescription')">{{ detailData.reasonDescription || '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.totalQuantity')">{{ detailData.totalQuantity ?? '-' }}</el-descriptions-item>
        <el-descriptions-item :label="t('rma.totalAmount')">{{ formatAmount(detailData.totalAmount) }}</el-descriptions-item>
      </el-descriptions>
      <el-table :data="detailData.items" border stripe class="detail-items-table">
        <el-table-column label="SKU Code" prop="skuCode" min-width="120" />
        <el-table-column label="SKU Name" prop="skuName" min-width="140" />
        <el-table-column label="Expected Qty" prop="expectedQuantity" min-width="100" />
        <el-table-column label="Unit Price" min-width="100">
          <template #default="{ row }">{{ formatAmount(row.unitPrice) }}</template>
        </el-table-column>
      </el-table>
    </template>
    <template #footer>
      <el-button @click="emit('update:visible', false)">{{ t('common.close') || 'Close' }}</el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.detail-items-table {
  margin-top: 16px;
}
@media (max-width: 768px) {
  :deep(.el-dialog) {
    width: 92% !important;
    max-width: 92% !important;
  }
}
</style>
