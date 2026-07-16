<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { Delete } from '@element-plus/icons-vue'
import { RETURN_TYPE } from './constants'

const { t } = useI18n()

const props = defineProps<{
  visible: boolean
  title: string
  loading: boolean
}>()

const emit = defineEmits<{
  'update:visible': [value: boolean]
  submit: [form: {
    orderNo: string
    returnType: string
    customerId: number | undefined
    customerName: string
    contactPhone: string
    reasonCode: string
    reasonDescription: string
    warehouseId: number | undefined
    items: Array<{ skuCode: string; skuName: string; expectedQuantity: number; unitPrice: number }>
  }]
}>()

const formData = ref({
  rmaNo: '',
  orderNo: '',
  returnType: RETURN_TYPE.REFUND,
  customerId: undefined as number | undefined,
  customerName: '',
  contactPhone: '',
  reasonCode: '',
  reasonDescription: '',
  warehouseId: undefined as number | undefined,
  status: 'PENDING',
  items: [] as Array<{
    skuCode: string
    skuName: string
    expectedQuantity: number
    unitPrice: number
  }>
})

const returnTypeOptions = [
  { label: t('rma.returnTypeREFUND'), value: RETURN_TYPE.REFUND },
  { label: t('rma.returnTypeREPLACEMENT'), value: RETURN_TYPE.REPLACEMENT },
  { label: t('rma.returnTypeREPAIR'), value: RETURN_TYPE.REPAIR }
]

watch(() => props.visible, (val) => {
  if (val) {
    formData.value = {
      rmaNo: '',
      orderNo: '',
      returnType: RETURN_TYPE.REFUND,
      customerId: undefined,
      customerName: '',
      contactPhone: '',
      reasonCode: '',
      reasonDescription: '',
      warehouseId: undefined,
      status: 'PENDING',
      items: []
    }
  }
})

const addItem = () => {
  formData.value.items.push({
    skuCode: '',
    skuName: '',
    expectedQuantity: 1,
    unitPrice: 0
  })
}

const removeItem = (index: number) => {
  formData.value.items.splice(index, 1)
}
</script>

<template>
  <el-dialog
    :model-value="props.visible"
    :title="props.title"
    width="700px"
    :close-on-click-modal="false"
    @update:model-value="emit('update:visible', $event)"
  >
    <el-form :model="formData" label-width="140px">
      <el-form-item :label="t('rma.orderNo')" required>
        <el-input v-model="formData.orderNo" :placeholder="t('common.placeholderSuffix') + t('rma.orderNo')" />
      </el-form-item>
      <el-form-item :label="t('rma.returnType')" required>
        <el-select v-model="formData.returnType" :placeholder="t('common.selectPlaceholder')">
          <el-option v-for="item in returnTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
        </el-select>
      </el-form-item>
      <el-form-item :label="t('rma.customerName')" required>
        <el-input v-model="formData.customerName" :placeholder="t('common.placeholderSuffix') + t('rma.customerName')" />
      </el-form-item>
      <el-form-item :label="t('rma.contactPhone')">
        <el-input v-model="formData.contactPhone" :placeholder="t('common.placeholderSuffix') + t('rma.contactPhone')" />
      </el-form-item>
      <el-form-item :label="t('rma.reasonCode')">
        <el-input v-model="formData.reasonCode" :placeholder="t('common.placeholderSuffix') + t('rma.reasonCode')" />
      </el-form-item>
      <el-form-item :label="t('rma.reasonDescription')">
        <el-input v-model="formData.reasonDescription" type="textarea" :rows="2" />
      </el-form-item>
      <el-form-item label="Items" required>
        <div class="items-wrapper">
          <div v-for="(item, index) in formData.items" :key="index" class="item-row">
            <el-input v-model="item.skuCode" placeholder="SKU Code" style="width: 120px" />
            <el-input v-model="item.skuName" placeholder="SKU Name" style="width: 140px" />
            <el-input-number v-model="item.expectedQuantity" :min="1" :max="99999" size="small" />
            <el-input-number v-model="item.unitPrice" :min="0" :precision="2" size="small" />
            <el-button type="danger" size="small" :icon="Delete" @click="removeItem(index)" />
          </div>
          <el-button type="primary" size="small" @click="addItem">+ Add Item</el-button>
        </div>
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="emit('update:visible', false)">{{ t('common.cancel') }}</el-button>
      <el-button type="primary" :loading="props.loading" @click="emit('submit', formData)">{{ t('common.submit') }}</el-button>
    </template>
  </el-dialog>
</template>
