<template>
  <div class="app-container">
    <el-card class="filter-card">
      <el-form :inline="true" class="demo-form-inline">
        <el-form-item label="Status">
          <el-select v-model="listQuery.status" placeholder="All" clearable @change="handleSearch">
            <el-option label="PENDING" value="PENDING" />
            <el-option label="APPROVED" value="APPROVED" />
            <el-option label="REJECTED" value="REJECTED" />
            <el-option label="DISABLED" value="DISABLED" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">Search</el-button>
          <el-button @click="handleReset">Reset</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <el-table v-loading="listLoading" :data="list" stripe border>
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="name" label="Name" min-width="140" />
        <el-table-column prop="code" label="Code" width="120" />
        <el-table-column label="Status" width="110">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="contactName" label="Contact" width="120" />
        <el-table-column prop="contactEmail" label="Email" min-width="180" />
        <el-table-column prop="contactPhone" label="Phone" width="130" />
        <el-table-column prop="createdAt" label="Created" width="170">
          <template #default="{ row }">{{ formatDate(row.createdAt) }}</template>
        </el-table-column>
        <el-table-column label="Actions" width="260" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="handleView(row)">View</el-button>
            <el-button v-if="row.status === 'PENDING'" type="success" size="small" @click="handleApprove(row)">Approve</el-button>
            <el-button v-if="row.status === 'PENDING'" type="warning" size="small" @click="handleReject(row)">Reject</el-button>
            <el-button v-if="row.status === 'APPROVED'" type="danger" size="small" @click="handleDisable(row)">Disable</el-button>
            <el-button type="info" size="small" @click="handleShop(row)">Shop</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        v-show="total > 0"
        v-model:page-size="listQuery.pageSize"
        v-model:current-page="listQuery.pageNum"
        :page-sizes="[10, 20, 50]"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </el-card>

    <el-dialog v-model="viewDialogVisible" title="Tenant Detail" width="600px">
      <el-descriptions v-if="currentTenant" :column="2" border>
        <el-descriptions-item label="ID" :span="1">{{ currentTenant.id }}</el-descriptions-item>
        <el-descriptions-item label="Status" :span="1">
          <el-tag :type="statusType(currentTenant.status)" size="small">{{ currentTenant.status }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Name" :span="2">{{ currentTenant.name }}</el-descriptions-item>
        <el-descriptions-item label="Code" :span="2">{{ currentTenant.code }}</el-descriptions-item>
        <el-descriptions-item label="Contact Name" :span="2">{{ currentTenant.contactName }}</el-descriptions-item>
        <el-descriptions-item label="Contact Email" :span="2">{{ currentTenant.contactEmail }}</el-descriptions-item>
        <el-descriptions-item label="Contact Phone" :span="2">{{ currentTenant.contactPhone }}</el-descriptions-item>
        <el-descriptions-item label="Domain" :span="2">{{ currentTenant.domain }}</el-descriptions-item>
        <el-descriptions-item label="Created" :span="1">{{ formatDate(currentTenant.createdAt) }}</el-descriptions-item>
        <el-descriptions-item label="Updated" :span="1">{{ formatDate(currentTenant.updatedAt) }}</el-descriptions-item>
      </el-descriptions>
    </el-dialog>

    <el-dialog v-model="shopDialogVisible" title="Shop Management" width="500px">
      <el-form v-if="currentShop" ref="shopFormRef" :model="currentShop" label-width="100px">
        <el-form-item label="Shop Name">
          <el-input v-model="currentShop.name" />
        </el-form-item>
        <el-form-item label="Description">
          <el-input v-model="currentShop.description" type="textarea" :rows="3" />
        </el-form-item>
        <el-form-item label="Logo URL">
          <el-input v-model="currentShop.logo" />
        </el-form-item>
        <el-form-item label="Banner URL">
          <el-input v-model="currentShop.banner" />
        </el-form-item>
        <el-form-item label="Status">
          <el-select v-model="currentShop.status">
            <el-option label="OPEN" value="OPEN" />
            <el-option label="CLOSED" value="CLOSED" />
          </el-select>
        </el-form-item>
        <el-form-item label="Sort Order">
          <el-input-number v-model="currentShop.sortOrder" :min="0" />
        </el-form-item>
      </el-form>
      <p v-else>No shop configured for this tenant.</p>
      <template #footer>
        <el-button @click="shopDialogVisible = false">Cancel</el-button>
        <el-button v-if="currentShop" type="primary" @click="handleSaveShop">Save</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  getTenantListAPI, getTenantByIdAPI,
  approveTenantAPI, rejectTenantAPI, disableTenantAPI,
  getShopByTenantAPI, updateShopAPI,
  type Tenant, type TenantShop, type TenantQueryParam,
} from '@/apis/tenant'

const listQuery = ref<TenantQueryParam>({ pageNum: 1, pageSize: 10 })
const list = ref<Tenant[]>([])
const listLoading = ref(true)
const total = ref(0)

const viewDialogVisible = ref(false)
const shopDialogVisible = ref(false)
const currentTenant = ref<Tenant | null>(null)
const currentShop = ref<TenantShop | null>(null)

const statusType = (status?: string) => {
  switch (status) {
    case 'APPROVED': return 'success'
    case 'PENDING': return 'warning'
    case 'REJECTED': return 'danger'
    case 'DISABLED': return 'info'
    default: return ''
  }
}

const formatDate = (date?: string) => {
  if (!date) return ''
  return date.replace('T', ' ').substring(0, 19)
}

const fetchList = async () => {
  listLoading.value = true
  try {
    const res = await getTenantListAPI(listQuery.value)
    list.value = res.data?.list || []
    total.value = res.data?.total || 0
  } finally {
    listLoading.value = false
  }
}

const handleSearch = () => {
  listQuery.value.pageNum = 1
  fetchList()
}

const handleReset = () => {
  listQuery.value = { pageNum: 1, pageSize: 10 }
  fetchList()
}

const handleView = async (row: Tenant) => {
  try {
    const res = await getTenantByIdAPI(row.id!)
    currentTenant.value = res.data
    viewDialogVisible.value = true
  } catch {
    ElMessage.error('Failed to fetch tenant details')
  }
}

const handleApprove = async (row: Tenant) => {
  try {
    await ElMessageBox.confirm(`Approve tenant "${row.name}"?`, 'Confirm')
    await approveTenantAPI(row.id!)
    ElMessage.success('Tenant approved')
    fetchList()
  } catch {
    // cancelled
  }
}

const handleReject = async (row: Tenant) => {
  try {
    await ElMessageBox.confirm(`Reject tenant "${row.name}"?`, 'Confirm')
    await rejectTenantAPI(row.id!)
    ElMessage.success('Tenant rejected')
    fetchList()
  } catch {
    // cancelled
  }
}

const handleDisable = async (row: Tenant) => {
  try {
    await ElMessageBox.confirm(`Disable tenant "${row.name}"?`, 'Confirm')
    await disableTenantAPI(row.id!)
    ElMessage.success('Tenant disabled')
    fetchList()
  } catch {
    // cancelled
  }
}

const handleShop = async (row: Tenant) => {
  try {
    const res = await getShopByTenantAPI(row.id!)
    currentShop.value = res.data || { tenantId: row.id, name: '', status: 'CLOSED', sortOrder: 0 } as TenantShop
    shopDialogVisible.value = true
  } catch {
    ElMessage.error('Failed to fetch shop info')
  }
}

const handleSaveShop = async () => {
  if (!currentShop.value) return
  try {
    await updateShopAPI(currentShop.value.tenantId!, currentShop.value)
    ElMessage.success('Shop updated')
    shopDialogVisible.value = false
  } catch {
    ElMessage.error('Failed to update shop')
  }
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageSize = val
  fetchList()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  fetchList()
}

onMounted(() => fetchList())
</script>

<style scoped>
.filter-card { margin-bottom: 16px; }
.table-card { margin-bottom: 16px; }
</style>
