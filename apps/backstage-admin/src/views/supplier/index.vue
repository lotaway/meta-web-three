<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, View, Check, Close } from '@element-plus/icons-vue'
import {
  getSupplierListAPI,
  getSupplierByIdAPI,
  createSupplierAPI,
  updateSupplierAPI,
  updateSupplierAssessmentAPI,
  type Supplier,
  type SupplierQueryParam
} from '@/apis/supplier'
import { formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const listQuery = ref<SupplierQueryParam>({
  pageNum: 1,
  pageSize: 10
})

const list = ref<Supplier[]>([])
const listLoading = ref(true)
const total = ref(0)

const dialogVisible = ref(false)
const dialogLoading = ref(false)
const isEdit = ref(false)
const viewMode = ref(false)

const formData = ref<Supplier>({
  supplierCode: '',
  supplierName: '',
  name: '',
  supplierType: '',
  province: '',
  city: '',
  address: '',
  contact: '',
  contactPerson: '',
  contactPhone: '',
  phone: '',
  email: '',
  status: 'ACTIVE',
  creditLimit: 0,
  paymentTerms: '',
  category: '',
  score: 0,
  level: '',
  assessmentLevel: ''
})

const statusOptions = [
  { label: 'Active', value: 'ACTIVE' },
  { label: 'Inactive', value: 'INACTIVE' },
  { label: 'Suspended', value: 'SUSPENDED' }
]

const supplierTypeOptions = [
  { label: 'Manufacturer', value: 'MANUFACTURER' },
  { label: 'Distributor', value: 'DISTRIBUTOR' },
  { label: 'Agent', value: 'AGENT' }
]

const categoryOptions = [
  { label: 'Raw Material', value: 'RAW_MATERIAL' },
  { label: 'Components', value: 'COMPONENTS' },
  { label: 'Packaging', value: 'PACKAGING' },
  { label: 'Equipment', value: 'EQUIPMENT' },
  { label: 'Services', value: 'SERVICES' }
]

const levelOptions = [
  { label: 'Level A', value: 'A' },
  { label: 'Level B', value: 'B' },
  { label: 'Level C', value: 'C' },
  { label: 'Level D', value: 'D' }
]

const getList = async () => {
  listLoading.value = true
  try {
    const response = await getSupplierListAPI(listQuery.value)
    listLoading.value = false
    list.value = response.data || []
    total.value = response.data?.length || 0
  } catch (error) {
    listLoading.value = false
    ElMessage.error('Failed to load suppliers')
  }
}

onMounted(() => {
  getList()
})

const handleSearch = () => {
  getList()
}

const handleReset = () => {
  listQuery.value = {
    pageNum: 1,
    pageSize: 10
  }
  getList()
}

const handleAdd = () => {
  isEdit.value = false
  viewMode.value = false
  formData.value = {
    supplierCode: '',
    supplierName: '',
    name: '',
    supplierType: '',
    province: '',
    city: '',
    address: '',
    contact: '',
    contactPerson: '',
    contactPhone: '',
    phone: '',
    email: '',
    status: 'ACTIVE',
    creditLimit: 0,
    paymentTerms: '',
    category: '',
    score: 0,
    level: '',
    assessmentLevel: ''
  }
  dialogVisible.value = true
}

const handleEdit = async (row: Supplier) => {
  if (!row.id) return
  try {
    const response = await getSupplierByIdAPI(row.id)
    formData.value = response.data || row
    isEdit.value = true
    viewMode.value = false
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load supplier details')
  }
}

const handleView = async (row: Supplier) => {
  if (!row.id) return
  try {
    const response = await getSupplierByIdAPI(row.id)
    formData.value = response.data || row
    isEdit.value = false
    viewMode.value = true
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error('Failed to load supplier details')
  }
}

const handleSubmit = async () => {
  dialogLoading.value = true
  try {
    if (isEdit.value && formData.value.id) {
      await updateSupplierAPI(formData.value.id, formData.value)
      ElMessage.success('Supplier updated successfully')
    } else {
      await createSupplierAPI(formData.value)
      ElMessage.success('Supplier created successfully')
    }
    dialogVisible.value = false
    getList()
  } catch (error) {
    ElMessage.error(isEdit.value ? 'Failed to update supplier' : 'Failed to create supplier')
  } finally {
    dialogLoading.value = false
  }
}

const handleAssessment = async (row: Supplier, level: string) => {
  if (!row.id) return
  try {
    await ElMessageBox.confirm(`Set supplier assessment level to ${level}?`, 'Confirm', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await updateSupplierAssessmentAPI(row.id, level)
    ElMessage.success('Assessment updated successfully')
    getList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('Failed to update assessment')
    }
  }
}

const getStatusType = (status: string): 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    'ACTIVE': 'success',
    'INACTIVE': 'info',
    'SUSPENDED': 'warning'
  }
  return map[status]
}

const getLevelType = (level: string): 'success' | 'warning' | 'info' | 'danger' | undefined => {
  const map: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
    'A': 'success',
    'B': 'warning',
    'C': 'info',
    'D': 'danger'
  }
  return map[level]
}
</script>

<template>
  <div class="supplier-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="listQuery" class="search-form">
        <el-form-item :label="t('supplier.status')">
          <el-select v-model="listQuery.status" :placeholder="t('supplier.statusPlaceholder')" clearable>
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('supplier.category')">
          <el-select v-model="listQuery.category" :placeholder="t('supplier.categoryPlaceholder')" clearable>
            <el-option v-for="item in categoryOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button :icon="Close" @click="handleReset">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="primary" :icon="Plus" @click="handleAdd">{{ t('common.add') }}</el-button>
      </div>

      <el-table v-loading="listLoading" :data="list" border stripe>
        <el-table-column prop="supplierCode" :label="t('supplier.code')" width="120" />
        <el-table-column prop="supplierName" :label="t('supplier.name')" min-width="150" />
        <el-table-column prop="supplierType" :label="t('supplier.type')" width="120" />
        <el-table-column prop="contactPerson" :label="t('supplier.contactPerson')" width="100" />
        <el-table-column prop="contactPhone" :label="t('supplier.phone')" width="130" />
        <el-table-column prop="province" :label="t('supplier.province')" width="100" />
        <el-table-column prop="city" :label="t('supplier.city')" width="100" />
        <el-table-column prop="status" :label="t('supplier.status')" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status || '')">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="level" :label="t('supplier.level')" width="100">
          <template #default="{ row }">
            <el-tag v-if="row.level" :type="getLevelType(row.level)">{{ row.level }}</el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="score" :label="t('supplier.score')" width="80" />
        <el-table-column :label="t('common.actions')" width="200" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">{{ t('common.view') }}</el-button>
            <el-button link type="primary" size="small" @click="handleEdit(row)">{{ t('common.edit') }}</el-button>
            <el-dropdown @command="(cmd: string) => handleAssessment(row, cmd)">
              <el-button link type="primary" size="small">{{ t('supplier.assess') }}</el-button>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="A">Level A</el-dropdown-item>
                  <el-dropdown-item command="B">Level B</el-dropdown-item>
                  <el-dropdown-item command="C">Level C</el-dropdown-item>
                  <el-dropdown-item command="D">Level D</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="viewMode ? t('supplier.detail') : (isEdit ? t('supplier.edit') : t('supplier.add'))"
      width="700px"
      :close-on-click-modal="false"
    >
      <el-form v-loading="dialogLoading" :model="formData" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('supplier.code')">
              <el-input v-model="formData.supplierCode" :disabled="isEdit || viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('supplier.name')">
              <el-input v-model="formData.supplierName" :disabled="viewMode" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('supplier.type')">
              <el-select v-model="formData.supplierType" :disabled="viewMode" style="width: 100%">
                <el-option v-for="item in supplierTypeOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('supplier.category')">
              <el-select v-model="formData.category" :disabled="viewMode" style="width: 100%">
                <el-option v-for="item in categoryOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('supplier.contactPerson')">
              <el-input v-model="formData.contactPerson" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('supplier.contactPhone')">
              <el-input v-model="formData.contactPhone" :disabled="viewMode" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('supplier.phone')">
              <el-input v-model="formData.phone" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('supplier.email')">
              <el-input v-model="formData.email" :disabled="viewMode" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item :label="t('supplier.province')">
              <el-input v-model="formData.province" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item :label="t('supplier.city')">
              <el-input v-model="formData.city" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item :label="t('supplier.address')">
              <el-input v-model="formData.address" :disabled="viewMode" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('supplier.status')">
              <el-select v-model="formData.status" :disabled="viewMode" style="width: 100%">
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('supplier.creditLimit')">
              <el-input-number v-model="formData.creditLimit" :min="0" :disabled="viewMode" style="width: 100%" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('supplier.paymentTerms')">
              <el-input v-model="formData.paymentTerms" :disabled="viewMode" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('supplier.level')">
              <el-select v-model="formData.level" :disabled="viewMode" style="width: 100%">
                <el-option v-for="item in levelOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button v-if="!viewMode" type="primary" :loading="dialogLoading" @click="handleSubmit">
          {{ t('common.submit') }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.supplier-container {
  padding: 20px;
}

.search-card {
  margin-bottom: 20px;
}

.table-card .toolbar {
  margin-bottom: 15px;
}
</style>