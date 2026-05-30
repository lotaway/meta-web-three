<template>
  <div class="exchange-rate-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>Exchange Rate Management</span>
          <el-button type="primary" @click="handleCreate">Add Rate</el-button>
        </div>
      </template>

      <el-table :data="rateList" border style="width: 100%">
        <el-table-column prop="sourceCurrency" label="Source Currency" width="150" />
        <el-table-column prop="targetCurrency" label="Target Currency" width="150" />
        <el-table-column prop="rate" label="Rate" width="120">
          <template #default="{ row }">
            {{ row.rate?.toFixed(4) }}
          </template>
        </el-table-column>
        <el-table-column prop="effectiveDate" label="Effective Date" width="150" />
        <el-table-column prop="rateType" label="Rate Type" width="120" />
        <el-table-column prop="isActive" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="row.isActive ? 'success' : 'info'">
              {{ row.isActive ? 'Active' : 'Inactive' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdBy" label="Created By" width="120" />
        <el-table-column label="Actions" fixed="right" width="180">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleEdit(row)">
              Edit
            </el-button>
            <el-button link type="danger" size="small" @click="handleDelete(row)">
              Delete
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="500px"
      @close="handleDialogClose"
    >
      <el-form ref="formRef" :model="formData" :rules="formRules" label-width="140px">
        <el-form-item label="Source Currency" prop="sourceCurrency">
          <el-select v-model="formData.sourceCurrency" placeholder="Select currency">
            <el-option label="USD" value="USD" />
            <el-option label="EUR" value="EUR" />
            <el-option label="GBP" value="GBP" />
            <el-option label="JPY" value="JPY" />
            <el-option label="CNY" value="CNY" />
            <el-option label="HKD" value="HKD" />
          </el-select>
        </el-form-item>
        <el-form-item label="Target Currency" prop="targetCurrency">
          <el-select v-model="formData.targetCurrency" placeholder="Select currency">
            <el-option label="CNY" value="CNY" />
            <el-option label="USD" value="USD" />
            <el-option label="EUR" value="EUR" />
            <el-option label="GBP" value="GBP" />
            <el-option label="JPY" value="JPY" />
            <el-option label="HKD" value="HKD" />
          </el-select>
        </el-form-item>
        <el-form-item label="Rate" prop="rate">
          <el-input-number v-model="formData.rate" :precision="6" :min="0" />
        </el-form-item>
        <el-form-item label="Effective Date" prop="effectiveDate">
          <el-date-picker
            v-model="formData.effectiveDate"
            type="date"
            placeholder="Select date"
            value-format="YYYY-MM-DD"
          />
        </el-form-item>
        <el-form-item label="Rate Type" prop="rateType">
          <el-select v-model="formData.rateType" placeholder="Select type">
            <el-option label="Spot" value="SPOT" />
            <el-option label="Middle" value="MIDDLE" />
            <el-option label="Selling" value="SELLING" />
            <el-option label="Buying" value="BUYING" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleSubmit">Confirm</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance } from 'element-plus'
import {
  getActiveRates,
  createExchangeRate,
  updateExchangeRate,
  deleteExchangeRate,
  type ExchangeRate,
  type CreateExchangeRateRequest
} from '@/apis/exchange'

const rateList = ref<ExchangeRate[]>([])
const dialogVisible = ref(false)
const dialogTitle = ref('Add Exchange Rate')
const formRef = ref<FormInstance>()
const isEdit = ref(false)
const editingId = ref<number>()

const formData = ref<CreateExchangeRateRequest>({
  sourceCurrency: '',
  targetCurrency: '',
  rate: 0,
  effectiveDate: '',
  rateType: 'SPOT',
  createdBy: 'admin'
})

const formRules = {
  sourceCurrency: [{ required: true, message: 'Please select source currency', trigger: 'change' }],
  targetCurrency: [{ required: true, message: 'Please select target currency', trigger: 'change' }],
  rate: [{ required: true, message: 'Please enter rate', trigger: 'blur' }],
  effectiveDate: [{ required: true, message: 'Please select effective date', trigger: 'change' }],
  rateType: [{ required: true, message: 'Please select rate type', trigger: 'change' }]
}

const loadRates = async () => {
  try {
    const res = await getActiveRates()
    rateList.value = res.data || []
  } catch (error) {
    ElMessage.error('Failed to load exchange rates')
  }
}

const handleCreate = () => {
  isEdit.value = false
  dialogTitle.value = 'Add Exchange Rate'
  formData.value = {
    sourceCurrency: '',
    targetCurrency: '',
    rate: 0,
    effectiveDate: '',
    rateType: 'SPOT',
    createdBy: 'admin'
  }
  dialogVisible.value = true
}

const handleEdit = (row: ExchangeRate) => {
  isEdit.value = true
  dialogTitle.value = 'Edit Exchange Rate'
  editingId.value = row.id
  formData.value = {
    sourceCurrency: row.sourceCurrency,
    targetCurrency: row.targetCurrency,
    rate: row.rate,
    effectiveDate: row.effectiveDate,
    rateType: row.rateType,
    createdBy: row.createdBy || 'admin'
  }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  if (!formRef.value) return
  await formRef.value.validate(async (valid) => {
    if (valid) {
      try {
        if (isEdit.value && editingId.value) {
          await updateExchangeRate(editingId.value, formData.value.rate)
          ElMessage.success('Exchange rate updated successfully')
        } else {
          await createExchangeRate(formData.value)
          ElMessage.success('Exchange rate created successfully')
        }
        dialogVisible.value = false
        loadRates()
      } catch (error) {
        ElMessage.error('Operation failed')
      }
    }
  })
}

const handleDelete = (row: ExchangeRate) => {
  if (!row.id) return
  ElMessageBox.confirm('Are you sure to delete this exchange rate?', 'Warning', {
    confirmButtonText: 'Confirm',
    cancelButtonText: 'Cancel',
    type: 'warning'
  }).then(async () => {
    try {
      await deleteExchangeRate(row.id!)
      ElMessage.success('Deleted successfully')
      loadRates()
    } catch (error) {
      ElMessage.error('Delete failed')
    }
  })
}

const handleDialogClose = () => {
  formRef.value?.resetFields()
}

onMounted(() => {
  loadRates()
})
</script>

<style scoped>
.exchange-rate-container {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>