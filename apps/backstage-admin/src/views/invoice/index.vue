<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Refresh } from '@element-plus/icons-vue'
import {
  getInvoiceListAPI,
  getInvoiceByIdAPI,
  createInvoiceAPI,
  issueInvoiceAPI,
  printInvoiceAPI,
  voidInvoiceAPI,
  redFlushInvoiceAPI,
  type InvoiceDTO,
  type InvoiceRequest,
} from '@/apis/invoice'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { formatDateTime } from '@/utils/datetime'

const loading = ref(false)
const list = ref<InvoiceDTO[]>([])
const total = ref(0)

const query = reactive({
  pageNum: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  status: undefined as string | undefined,
  customerId: undefined as number | undefined,
  startDate: undefined as string | undefined,
  endDate: undefined as string | undefined,
})

const dialogVisible = ref(false)
const dialogTitle = ref('')
const detailVisible = ref(false)
const detail = ref<InvoiceDTO>({})

const form = reactive<InvoiceRequest>({
  invoiceNo: '',
  orderNo: '',
  customerId: 0,
  customerName: '',
  customerTaxNo: '',
  type: 'NORMAL',
  amount: 0,
  taxRate: '13',
})

const statusOptions = [
  { label: 'Draft', value: 'DRAFT' },
  { label: 'Issued', value: 'ISSUED' },
  { label: 'Printed', value: 'PRINTED' },
  { label: 'Voided', value: 'VOIDED' },
  { label: 'Red Flushed', value: 'RED_FLUSHED' },
]

const getList = async () => {
  loading.value = true
  try {
    const res = await getInvoiceListAPI(query)
    list.value = res.data.list
    total.value = res.data.total
  } catch (error) {
    console.error('Failed to load invoice list:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => { getList() })

const handleSearch = () => {
  query.pageNum = 1
  getList()
}

const handleReset = () => {
  query.status = undefined
  query.customerId = undefined
  query.startDate = undefined
  query.endDate = undefined
  query.pageNum = 1
  getList()
}

const handleAdd = () => {
  dialogTitle.value = 'Create Invoice'
  form.invoiceNo = ''
  form.orderNo = ''
  form.customerId = 0
  form.customerName = ''
  form.customerTaxNo = ''
  form.type = 'NORMAL'
  form.amount = 0
  form.taxRate = '13'
  dialogVisible.value = true
}

const handleCreate = async () => {
  try {
    await createInvoiceAPI(form)
    ElMessage.success({ message: 'Invoice created', duration: MESSAGE_DURATION_SHORT })
    dialogVisible.value = false
    getList()
  } catch (error) {
    console.error('Failed to create invoice:', error)
  }
}

const handleView = async (row: InvoiceDTO) => {
  if (!row.id) return
  try {
    const res = await getInvoiceByIdAPI(row.id)
    detail.value = res.data
    detailVisible.value = true
  } catch (error) {
    console.error('Failed to load invoice:', error)
  }
}

const handleIssue = async (row: InvoiceDTO) => {
  if (!row.id) return
  try {
    await ElMessageBox.prompt('Enter issuer name:', 'Issue Invoice', {
      confirmButtonText: 'Submit', cancelButtonText: 'Cancel', inputPlaceholder: 'Issuer name',
    }).then(async ({ value }) => {
      if (!value) return
      await issueInvoiceAPI(row.id!, value)
      ElMessage.success({ message: 'Invoice issued', duration: MESSAGE_DURATION_SHORT })
      getList()
    })
  } catch (error) {
    if (error !== 'cancel') console.error('Failed to issue invoice:', error)
  }
}

const handlePrint = async (row: InvoiceDTO) => {
  if (!row.id) return
  try {
    await ElMessageBox.confirm('Print this invoice?', 'Confirm', {
      confirmButtonText: 'Print', cancelButtonText: 'Cancel', type: 'info',
    })
    await printInvoiceAPI(row.id)
    ElMessage.success({ message: 'Invoice printed', duration: MESSAGE_DURATION_SHORT })
    getList()
  } catch (error) {
    if (error !== 'cancel') console.error('Failed to print invoice:', error)
  }
}

const handleVoid = async (row: InvoiceDTO) => {
  if (!row.id) return
  try {
    await ElMessageBox.prompt('Enter void reason:', 'Void Invoice', {
      confirmButtonText: 'Submit', cancelButtonText: 'Cancel', inputPlaceholder: 'Reason',
    }).then(async ({ value }) => {
      if (!value) return
      await voidInvoiceAPI(row.id!, value)
      ElMessage.success({ message: 'Invoice voided', duration: MESSAGE_DURATION_SHORT })
      getList()
    })
  } catch (error) {
    if (error !== 'cancel') console.error('Failed to void invoice:', error)
  }
}

const handleRedFlush = async (row: InvoiceDTO) => {
  if (!row.id) return
  try {
    await ElMessageBox.prompt('Enter red flush reason:', 'Red Flush Invoice', {
      confirmButtonText: 'Submit', cancelButtonText: 'Cancel', inputPlaceholder: 'Reason',
    }).then(async ({ value }) => {
      if (!value) return
      await redFlushInvoiceAPI(row.id!, value)
      ElMessage.success({ message: 'Red flush completed', duration: MESSAGE_DURATION_SHORT })
      getList()
    })
  } catch (error) {
    if (error !== 'cancel') console.error('Failed to red flush invoice:', error)
  }
}

const getStatusTag = (status?: string): 'info' | 'success' | 'warning' | 'danger' | 'primary' => {
  const map: Record<string, 'info' | 'success' | 'warning' | 'danger' | 'primary'> = {
    DRAFT: 'info', ISSUED: 'success', PRINTED: 'primary', VOIDED: 'danger', RED_FLUSHED: 'warning',
  }
  return map[status || ''] || 'info'
}
</script>

<template>
  <div class="invoice-container">
    <el-card class="search-card">
      <el-form :inline="true" :model="query">
        <el-form-item label="Status">
          <el-select v-model="query.status" placeholder="Select status" clearable style="width: 140px">
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="Customer ID">
          <el-input-number v-model="query.customerId" :min="1" placeholder="Customer ID" clearable />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :icon="Search" @click="handleSearch">Search</el-button>
          <el-button :icon="Refresh" @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="toolbar">
        <el-button type="primary" :icon="Plus" @click="handleAdd">Create Invoice</el-button>
      </div>

      <el-table v-loading="loading" :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="invoiceNo" label="Invoice No" width="160" />
        <el-table-column prop="orderNo" label="Order No" width="160" />
        <el-table-column prop="customerName" label="Customer" min-width="120" />
        <el-table-column prop="amount" label="Amount" width="100">
          <template #default="{ row }">¥{{ row.amount?.toFixed(2) }}</template>
        </el-table-column>
        <el-table-column prop="taxRate" label="Tax Rate" width="80">
          <template #default="{ row }">{{ row.taxRate }}%</template>
        </el-table-column>
        <el-table-column prop="status" label="Status" width="110">
          <template #default="{ row }">
            <el-tag :type="getStatusTag(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="Created" width="170">
          <template #default="{ row }">{{ row.createTime ? formatDateTime(row.createTime) : '-' }}</template>
        </el-table-column>
        <el-table-column label="Actions" width="310" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="handleView(row)">View</el-button>
            <el-button v-if="row.status === 'DRAFT'" link type="success" size="small" @click="handleIssue(row)">Issue</el-button>
            <el-button v-if="row.status === 'ISSUED'" link type="primary" size="small" @click="handlePrint(row)">Print</el-button>
            <el-button v-if="row.status === 'DRAFT' || row.status === 'ISSUED'" link type="danger" size="small" @click="handleVoid(row)">Void</el-button>
            <el-button v-if="row.status === 'ISSUED' || row.status === 'PRINTED'" link type="warning" size="small" @click="handleRedFlush(row)">Red Flush</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        v-model:current-page="query.pageNum"
        v-model:page-size="query.pageSize"
        :page-sizes="PAGE_SIZE_OPTIONS"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="getList"
        @current-change="getList"
        class="pagination"
      />
    </el-card>

    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="550px" :close-on-click-modal="false">
      <el-form :model="form" label-width="130px">
        <el-form-item label="Invoice No" prop="invoiceNo">
          <el-input v-model="form.invoiceNo" placeholder="Enter invoice number" />
        </el-form-item>
        <el-form-item label="Order No" prop="orderNo">
          <el-input v-model="form.orderNo" placeholder="Enter order number" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Customer ID" prop="customerId">
              <el-input-number v-model="form.customerId" :min="1" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Type" prop="type">
              <el-select v-model="form.type" style="width: 100%">
                <el-option label="Normal" value="NORMAL" />
                <el-option label="Simplified" value="SIMPLIFIED" />
                <el-option label="Special" value="SPECIAL" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="Customer Name" prop="customerName">
          <el-input v-model="form.customerName" placeholder="Enter customer name" />
        </el-form-item>
        <el-form-item label="Customer Tax No" prop="customerTaxNo">
          <el-input v-model="form.customerTaxNo" placeholder="Enter tax number" />
        </el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="Amount" prop="amount">
              <el-input-number v-model="form.amount" :min="0" :precision="2" style="width: 100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Tax Rate (%)" prop="taxRate">
              <el-select v-model="form.taxRate" style="width: 100%">
                <el-option label="0%" value="0" />
                <el-option label="3%" value="3" />
                <el-option label="6%" value="6" />
                <el-option label="9%" value="9" />
                <el-option label="13%" value="13" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleCreate">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="detailVisible" title="Invoice Detail" width="600px">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="ID">{{ detail.id }}</el-descriptions-item>
        <el-descriptions-item label="Invoice No">{{ detail.invoiceNo }}</el-descriptions-item>
        <el-descriptions-item label="Order No">{{ detail.orderNo }}</el-descriptions-item>
        <el-descriptions-item label="Customer">{{ detail.customerName }}</el-descriptions-item>
        <el-descriptions-item label="Tax No">{{ detail.customerTaxNo }}</el-descriptions-item>
        <el-descriptions-item label="Type">{{ detail.type }}</el-descriptions-item>
        <el-descriptions-item label="Amount">¥{{ detail.amount?.toFixed(2) }}</el-descriptions-item>
        <el-descriptions-item label="Tax Rate">{{ detail.taxRate }}%</el-descriptions-item>
        <el-descriptions-item label="Status">
          <el-tag :type="getStatusTag(detail.status)">{{ detail.status }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Issuer">{{ detail.issuer || '-' }}</el-descriptions-item>
        <el-descriptions-item label="Issue Date">{{ detail.issueDate ? formatDateTime(detail.issueDate) : '-' }}</el-descriptions-item>
        <el-descriptions-item label="Print Date">{{ detail.printDate ? formatDateTime(detail.printDate) : '-' }}</el-descriptions-item>
        <el-descriptions-item label="Void Reason" :span="2">{{ detail.voidReason || '-' }}</el-descriptions-item>
        <el-descriptions-item label="Red Flush Reason" :span="2">{{ detail.redFlushReason || '-' }}</el-descriptions-item>
      </el-descriptions>
      <template #footer>
        <el-button @click="detailVisible = false">Close</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.invoice-container {
  padding: 20px;
}
.search-card {
  margin-bottom: 20px;
}
.table-card {
  margin-bottom: 20px;
}
.toolbar {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
}
.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>
