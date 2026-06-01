<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Edit, Delete, House, Box, List, Document } from '@element-plus/icons-vue'
import {
  getWarehouseListAPI,
  createWarehouseAPI,
  updateWarehouseAPI,
  deleteWarehouseAPI,
  getInboundOrderListAPI,
  getOutboundOrderListAPI,
  getInventoryListAPI
} from '@/apis/warehouse'
import type { WarehouseDTO, InboundOrderDTO, OutboundOrderDTO, InventoryDTO } from '@/apis/warehouse'
import { MESSAGE_DURATION_SHORT, DEFAULT_PAGE_SIZE, PAGE_SIZE_OPTIONS } from '@/constants'
import { t } from '@/locales'

const activeTab = ref('warehouse')

// Warehouse state
const warehouseQuery = ref<{ pageNum: number; pageSize: number; status?: string }>({ pageNum: 1, pageSize: DEFAULT_PAGE_SIZE, status: undefined })
const warehouseList = ref<WarehouseDTO[]>([])
const warehouseTotal = ref(0)
const warehouseLoading = ref(false)
const warehouseDialogVisible = ref(false)
const warehouseDialogTitle = ref('')
const warehouseForm = reactive<WarehouseDTO>({
  warehouseCode: '',
  warehouseName: '',
  address: '',
  province: '',
  city: '',
  district: '',
  contactName: '',
  contactPhone: '',
  managerName: '',
  capacity: 0,
  status: 'ACTIVE'
})
const warehouseFormRef = ref()

// Inbound order state
const inboundQuery = ref({ pageNum: 1, pageSize: DEFAULT_PAGE_SIZE, warehouseId: undefined as number | undefined, status: undefined as string | undefined })
const inboundList = ref<InboundOrderDTO[]>([])
const inboundTotal = ref(0)
const inboundLoading = ref(false)

// Outbound order state
const outboundQuery = ref({ pageNum: 1, pageSize: DEFAULT_PAGE_SIZE, warehouseId: undefined as number | undefined, status: undefined as string | undefined })
const outboundList = ref<OutboundOrderDTO[]>([])
const outboundTotal = ref(0)
const outboundLoading = ref(false)

// Inventory state
const inventoryQuery = ref({ pageNum: 1, pageSize: DEFAULT_PAGE_SIZE, warehouseId: undefined as number | undefined, productName: '', skuCode: '' })
const inventoryList = ref<InventoryDTO[]>([])
const inventoryTotal = ref(0)
const inventoryLoading = ref(false)

// Warehouse methods
const getWarehouseList = async () => {
  warehouseLoading.value = true
  try {
    const response = await getWarehouseListAPI(warehouseQuery.value)
    warehouseList.value = response.data.list
    warehouseTotal.value = response.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    warehouseLoading.value = false
  }
}

const handleWarehouseSearch = () => {
  warehouseQuery.value.pageNum = 1
  getWarehouseList()
}

const handleWarehouseReset = () => {
  warehouseQuery.value = { pageNum: 1, pageSize: DEFAULT_PAGE_SIZE }
  getWarehouseList()
}

const handleWarehouseAdd = () => {
  warehouseDialogTitle.value = t('warehouse.addWarehouse')
  Object.assign(warehouseForm, {
    id: undefined,
    warehouseCode: '',
    warehouseName: '',
    address: '',
    province: '',
    city: '',
    district: '',
    contactName: '',
    contactPhone: '',
    managerName: '',
    capacity: 0,
    status: 'ACTIVE'
  })
  warehouseDialogVisible.value = true
}

const handleWarehouseEdit = (row: WarehouseDTO) => {
  warehouseDialogTitle.value = t('warehouse.editWarehouse')
  Object.assign(warehouseForm, { ...row })
  warehouseDialogVisible.value = true
}

const handleWarehouseDelete = async (row: WarehouseDTO) => {
  try {
    await ElMessageBox.confirm(t('warehouse.confirmDelete'), t('common.warning'), {
      type: 'warning'
    })
    await deleteWarehouseAPI(row.id!)
    ElMessage.success(t('common.deleteSuccess'))
    getWarehouseList()
  } catch {
    // cancelled
  }
}

const handleWarehouseSubmit = async () => {
  try {
    if (warehouseForm.id) {
      await updateWarehouseAPI(warehouseForm.id, warehouseForm)
      ElMessage.success(t('common.updateSuccess'))
    } else {
      await createWarehouseAPI(warehouseForm)
      ElMessage.success(t('common.createSuccess'))
    }
    warehouseDialogVisible.value = false
    getWarehouseList()
  } catch {
    ElMessage.error(t('common.operationFailed'))
  }
}

const handleWarehouseSizeChange = (val: number) => {
  warehouseQuery.value.pageSize = val
  getWarehouseList()
}

const handleWarehouseCurrentChange = (val: number) => {
  warehouseQuery.value.pageNum = val
  getWarehouseList()
}

// Inbound order methods
const getInboundList = async () => {
  inboundLoading.value = true
  try {
    const response = await getInboundOrderListAPI(inboundQuery.value)
    inboundList.value = response.data.list
    inboundTotal.value = response.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    inboundLoading.value = false
  }
}

const handleInboundSearch = () => {
  inboundQuery.value.pageNum = 1
  getInboundList()
}

const handleInboundSizeChange = (val: number) => {
  inboundQuery.value.pageSize = val
  getInboundList()
}

const handleInboundCurrentChange = (val: number) => {
  inboundQuery.value.pageNum = val
  getInboundList()
}

// Outbound order methods
const getOutboundList = async () => {
  outboundLoading.value = true
  try {
    const response = await getOutboundOrderListAPI(outboundQuery.value)
    outboundList.value = response.data.list
    outboundTotal.value = response.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    outboundLoading.value = false
  }
}

const handleOutboundSearch = () => {
  outboundQuery.value.pageNum = 1
  getOutboundList()
}

const handleOutboundSizeChange = (val: number) => {
  outboundQuery.value.pageSize = val
  getOutboundList()
}

const handleOutboundCurrentChange = (val: number) => {
  outboundQuery.value.pageNum = val
  getOutboundList()
}

// Inventory methods
const getInventoryList = async () => {
  inventoryLoading.value = true
  try {
    const response = await getInventoryListAPI(inventoryQuery.value)
    inventoryList.value = response.data.list
    inventoryTotal.value = response.data.total
  } catch {
    ElMessage.error(t('common.queryFailed'))
  } finally {
    inventoryLoading.value = false
  }
}

const handleInventorySearch = () => {
  inventoryQuery.value.pageNum = 1
  getInventoryList()
}

const handleInventoryReset = () => {
  inventoryQuery.value = { pageNum: 1, pageSize: DEFAULT_PAGE_SIZE, warehouseId: undefined, productName: '', skuCode: '' }
  getInventoryList()
}

const handleInventorySizeChange = (val: number) => {
  inventoryQuery.value.pageSize = val
  getInventoryList()
}

const handleInventoryCurrentChange = (val: number) => {
  inventoryQuery.value.pageNum = val
  getInventoryList()
}

const statusOptions = [
  { value: 'ACTIVE', label: t('warehouse.statusActive') },
  { value: 'INACTIVE', label: t('warehouse.statusInactive') },
  { value: 'MAINTENANCE', label: t('warehouse.statusMaintenance') }
]

const inboundStatusOptions = [
  { value: 'PENDING', label: t('warehouse.inboundPending') },
  { value: 'IN_TRANSIT', label: t('warehouse.inboundInTransit') },
  { value: 'RECEIVED', label: t('warehouse.inboundReceived') },
  { value: 'REJECTED', label: t('warehouse.inboundRejected') },
  { value: 'CANCELLED', label: t('warehouse.inboundCancelled') }
]

const outboundStatusOptions = [
  { value: 'PENDING', label: t('warehouse.outboundPending') },
  { value: 'PICKING', label: t('warehouse.outboundPicking') },
  { value: 'SHIPPED', label: t('warehouse.outboundShipped') },
  { value: 'DELIVERED', label: t('warehouse.outboundDelivered') },
  { value: 'CANCELLED', label: t('warehouse.outboundCancelled') }
]

onMounted(() => {
  getWarehouseList()
})
</script>

<template>
  <div class="warehouse-container">
    <el-tabs v-model="activeTab" type="border-card">
      <el-tab-pane :label="t('warehouse.warehouseManagement')" name="warehouse">
        <div class="toolbar">
          <el-form :inline="true" :model="warehouseQuery">
            <el-form-item :label="t('warehouse.warehouseStatus')">
              <el-select v-model="warehouseQuery.status" :placeholder="t('common.select')" clearable>
                <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleWarehouseSearch">{{ t('common.search') }}</el-button>
              <el-button @click="handleWarehouseReset">{{ t('common.reset') }}</el-button>
              <el-button type="primary" :icon="Plus" @click="handleWarehouseAdd">{{ t('common.add') }}</el-button>
            </el-form-item>
          </el-form>
        </div>

        <el-table v-loading="warehouseLoading" :data="warehouseList" border stripe>
          <el-table-column :label="t('warehouse.id')" prop="id" width="80" />
          <el-table-column :label="t('warehouse.warehouseCode')" prop="warehouseCode" width="120" />
          <el-table-column :label="t('warehouse.warehouseName')" prop="warehouseName" min-width="150" />
          <el-table-column :label="t('warehouse.address')" prop="address" min-width="200" />
          <el-table-column :label="t('warehouse.contactName')" prop="contactName" width="100" />
          <el-table-column :label="t('warehouse.contactPhone')" prop="contactPhone" width="120" />
          <el-table-column :label="t('warehouse.managerName')" prop="managerName" width="100" />
          <el-table-column :label="t('warehouse.capacity')" prop="capacity" width="100">
            <template #default="{ row }">
              {{ row.capacity || 0 }}
            </template>
          </el-table-column>
          <el-table-column :label="t('warehouse.status')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="row.status === 'ACTIVE' ? 'success' : row.status === 'MAINTENANCE' ? 'warning' : 'info'">
                {{ statusOptions.find(o => o.value === row.status)?.label || row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('common.action')" width="150" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link :icon="Edit" @click="handleWarehouseEdit(row)">{{ t('common.edit') }}</el-button>
              <el-button type="danger" link :icon="Delete" @click="handleWarehouseDelete(row)">{{ t('common.delete') }}</el-button>
            </template>
          </el-table-column>
        </el-table>

        <el-pagination
          v-model:current-page="warehouseQuery.pageNum"
          v-model:page-size="warehouseQuery.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="warehouseTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleWarehouseSizeChange"
          @current-change="handleWarehouseCurrentChange"
        />
      </el-tab-pane>

      <el-tab-pane :label="t('warehouse.inboundManagement')" name="inbound">
        <div class="toolbar">
          <el-form :inline="true" :model="inboundQuery">
            <el-form-item :label="t('warehouse.warehouse')">
              <el-input v-model="inboundQuery.warehouseId" :placeholder="t('warehouse.warehouseId')" clearable type="number" />
            </el-form-item>
            <el-form-item :label="t('warehouse.status')">
              <el-select v-model="inboundQuery.status" :placeholder="t('common.select')" clearable>
                <el-option v-for="item in inboundStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleInboundSearch">{{ t('common.search') }}</el-button>
            </el-form-item>
          </el-form>
        </div>

        <el-table v-loading="inboundLoading" :data="inboundList" border stripe>
          <el-table-column :label="t('warehouse.orderNo')" prop="orderNo" width="180" />
          <el-table-column :label="t('warehouse.warehouseName')" prop="warehouseName" width="120" />
          <el-table-column :label="t('warehouse.supplierName')" prop="supplierName" width="150" />
          <el-table-column :label="t('warehouse.inboundType')" prop="inboundType" width="100" />
          <el-table-column :label="t('warehouse.totalQuantity')" prop="totalQuantity" width="100" />
          <el-table-column :label="t('warehouse.receivedQuantity')" prop="receivedQuantity" width="120" />
          <el-table-column :label="t('warehouse.expectedArrivalTime')" prop="expectedArrivalTime" width="160" />
          <el-table-column :label="t('warehouse.status')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="row.status === 'RECEIVED' ? 'success' : row.status === 'PENDING' ? 'info' : row.status === 'REJECTED' ? 'danger' : 'warning'">
                {{ inboundStatusOptions.find(o => o.value === row.status)?.label || row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('warehouse.createTime')" prop="createTime" width="160" />
        </el-table>

        <el-pagination
          v-model:current-page="inboundQuery.pageNum"
          v-model:page-size="inboundQuery.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="inboundTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleInboundSizeChange"
          @current-change="handleInboundCurrentChange"
        />
      </el-tab-pane>

      <el-tab-pane :label="t('warehouse.outboundManagement')" name="outbound">
        <div class="toolbar">
          <el-form :inline="true" :model="outboundQuery">
            <el-form-item :label="t('warehouse.warehouse')">
              <el-input v-model="outboundQuery.warehouseId" :placeholder="t('warehouse.warehouseId')" clearable type="number" />
            </el-form-item>
            <el-form-item :label="t('warehouse.status')">
              <el-select v-model="outboundQuery.status" :placeholder="t('common.select')" clearable>
                <el-option v-for="item in outboundStatusOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleOutboundSearch">{{ t('common.search') }}</el-button>
            </el-form-item>
          </el-form>
        </div>

        <el-table v-loading="outboundLoading" :data="outboundList" border stripe>
          <el-table-column :label="t('warehouse.orderNo')" prop="orderNo" width="180" />
          <el-table-column :label="t('warehouse.warehouseName')" prop="warehouseName" width="120" />
          <el-table-column :label="t('warehouse.orderType')" prop="orderType" width="100" />
          <el-table-column :label="t('warehouse.relatedOrderNo')" prop="relatedOrderNo" width="180" />
          <el-table-column :label="t('warehouse.totalQuantity')" prop="totalQuantity" width="100" />
          <el-table-column :label="t('warehouse.shippedQuantity')" prop="shippedQuantity" width="120" />
          <el-table-column :label="t('warehouse.expectedDeliveryTime')" prop="expectedDeliveryTime" width="160" />
          <el-table-column :label="t('warehouse.status')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="row.status === 'DELIVERED' ? 'success' : row.status === 'PENDING' ? 'info' : row.status === 'CANCELLED' ? 'danger' : 'warning'">
                {{ outboundStatusOptions.find(o => o.value === row.status)?.label || row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('warehouse.createTime')" prop="createTime" width="160" />
        </el-table>

        <el-pagination
          v-model:current-page="outboundQuery.pageNum"
          v-model:page-size="outboundQuery.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="outboundTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleOutboundSizeChange"
          @current-change="handleOutboundCurrentChange"
        />
      </el-tab-pane>

      <el-tab-pane :label="t('warehouse.inventoryManagement')" name="inventory">
        <div class="toolbar">
          <el-form :inline="true" :model="inventoryQuery">
            <el-form-item :label="t('warehouse.warehouse')">
              <el-input v-model="inventoryQuery.warehouseId" :placeholder="t('warehouse.warehouseId')" clearable type="number" />
            </el-form-item>
            <el-form-item :label="t('warehouse.productName')">
              <el-input v-model="inventoryQuery.productName" :placeholder="t('warehouse.productNamePlaceholder')" clearable />
            </el-form-item>
            <el-form-item :label="t('warehouse.skuCode')">
              <el-input v-model="inventoryQuery.skuCode" :placeholder="t('warehouse.skuCodePlaceholder')" clearable />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :icon="Search" @click="handleInventorySearch">{{ t('common.search') }}</el-button>
              <el-button @click="handleInventoryReset">{{ t('common.reset') }}</el-button>
            </el-form-item>
          </el-form>
        </div>

        <el-table v-loading="inventoryLoading" :data="inventoryList" border stripe>
          <el-table-column :label="t('warehouse.id')" prop="id" width="80" />
          <el-table-column :label="t('warehouse.warehouseName')" prop="warehouseName" width="120" />
          <el-table-column :label="t('warehouse.productName')" prop="productName" min-width="150" />
          <el-table-column :label="t('warehouse.productCode')" prop="productCode" width="120" />
          <el-table-column :label="t('warehouse.skuCode')" prop="skuCode" width="120" />
          <el-table-column :label="t('warehouse.quantity')" prop="quantity" width="100" />
          <el-table-column :label="t('warehouse.availableQuantity')" prop="availableQuantity" width="120" />
          <el-table-column :label="t('warehouse.lockedQuantity')" prop="lockedQuantity" width="120" />
          <el-table-column :label="t('warehouse.damagedQuantity')" prop="damagedQuantity" width="120" />
          <el-table-column :label="t('warehouse.lastInboundTime')" prop="lastInboundTime" width="160" />
          <el-table-column :label="t('warehouse.lastOutboundTime')" prop="lastOutboundTime" width="160" />
        </el-table>

        <el-pagination
          v-model:current-page="inventoryQuery.pageNum"
          v-model:page-size="inventoryQuery.pageSize"
          :page-sizes="PAGE_SIZE_OPTIONS"
          :total="inventoryTotal"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleInventorySizeChange"
          @current-change="handleInventoryCurrentChange"
        />
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="warehouseDialogVisible" :title="warehouseDialogTitle" width="600px">
      <el-form ref="warehouseFormRef" :model="warehouseForm" label-width="120px">
        <el-form-item :label="t('warehouse.warehouseCode')" prop="warehouseCode" :rules="[{ required: true, message: t('warehouse.codeRequired'), trigger: 'blur' }]">
          <el-input v-model="warehouseForm.warehouseCode" :placeholder="t('warehouse.warehouseCodePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('warehouse.warehouseName')" prop="warehouseName" :rules="[{ required: true, message: t('warehouse.nameRequired'), trigger: 'blur' }]">
          <el-input v-model="warehouseForm.warehouseName" :placeholder="t('warehouse.warehouseNamePlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('warehouse.address')" prop="address">
          <el-input v-model="warehouseForm.address" :placeholder="t('warehouse.addressPlaceholder')" />
        </el-form-item>
        <el-form-item :label="t('warehouse.contactName')" prop="contactName">
          <el-input v-model="warehouseForm.contactName" />
        </el-form-item>
        <el-form-item :label="t('warehouse.contactPhone')" prop="contactPhone">
          <el-input v-model="warehouseForm.contactPhone" />
        </el-form-item>
        <el-form-item :label="t('warehouse.managerName')" prop="managerName">
          <el-input v-model="warehouseForm.managerName" />
        </el-form-item>
        <el-form-item :label="t('warehouse.capacity')" prop="capacity">
          <el-input-number v-model="warehouseForm.capacity" :min="0" />
        </el-form-item>
        <el-form-item :label="t('warehouse.status')" prop="status">
          <el-select v-model="warehouseForm.status">
            <el-option v-for="item in statusOptions" :key="item.value" :label="item.label" :value="item.value" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="warehouseDialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleWarehouseSubmit">{{ t('common.submit') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.warehouse-container {
  padding: 20px;
}

.toolbar {
  margin-bottom: 20px;
}

.el-table {
  margin-top: 20px;
}

.el-pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>