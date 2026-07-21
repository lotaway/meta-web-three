<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh, Plus } from '@element-plus/icons-vue'
import {
  listOrdersAPI, createOrderAPI, scheduleOrderAPI,
  startProductionAPI, pauseProductionAPI, resumeProductionAPI,
  completeProductionAPI, cancelOrderAPI,
  listStationsAPI, createStationAPI, assignOrderToStationAPI,
  getStationBindingsAPI, bindEquipmentAPI, bindToolAPI, bindPersonnelAPI,
  unbindStationAPI,
  type ProductionOrder, type WorkStation, type WorkStationBinding,
} from '@/apis/production'

const activeTab = ref('orders')
const loading = ref(false)

// Orders
const orders = ref<ProductionOrder[]>([])
const loadOrders = async () => {
  try { const res = await listOrdersAPI(); orders.value = (res.data as any) || [] } catch (_) { /* ignore */ }
}

const orderDialog = ref(false)
const orderForm = ref({ productCode: '', productName: '', quantity: 10, priority: 'NORMAL', workshopCode: '' })
const createOrder = async () => {
  try {
    await createOrderAPI({ productCode: orderForm.value.productCode, productName: orderForm.value.productName, quantityPlanned: orderForm.value.quantity, priority: orderForm.value.priority as any, workshopCode: orderForm.value.workshopCode })
    ElMessage.success('Order created')
    orderDialog.value = false
    loadOrders()
  } catch (_) { ElMessage.error('Failed') }
}

const scheduleDialog = ref(false)
const scheduleTarget = ref<ProductionOrder | null>(null)
const scheduleLineCode = ref('')
const openSchedule = async (order: ProductionOrder) => {
  const { value } = await ElMessageBox.prompt('Enter production line code:', 'Schedule Order')
  if (!value) return
  try {
    await scheduleOrderAPI(order.id, value)
    ElMessage.success('Scheduled')
    loadOrders()
  } catch (_) { ElMessage.error('Failed') }
}

const lifecycleAction = async (id: number, action: string, fn: (id: number) => Promise<any>) => {
  try {
    await ElMessageBox.confirm(`${action} this order?`, 'Confirm', { type: 'warning' })
    await fn(id)
    ElMessage.success(`${action} successful`)
    loadOrders()
  } catch (_) { /* ignore */ }
}

// Stations
const stations = ref<WorkStation[]>([])
const loadStations = async () => {
  try { const res = await listStationsAPI(); stations.value = (res.data as any) || [] } catch (_) { /* ignore */ }
}

const stationDialog = ref(false)
const stationForm = ref({ stationCode: '', stationName: '', stationType: 'ASSEMBLY', workshopCode: '', capacity: 10 })
const createStation = async () => {
  try {
    await createStationAPI(stationForm.value)
    ElMessage.success('Station created')
    stationDialog.value = false
    loadStations()
  } catch (_) { ElMessage.error('Failed') }
}

const assignDialog = ref(false)
const assignStationCode = ref('')
const assignOrderCode = ref('')
const assignStation = async () => {
  try {
    await assignOrderToStationAPI(assignStationCode.value, assignOrderCode.value)
    ElMessage.success('Assigned')
    assignDialog.value = false
    loadStations()
  } catch (_) { ElMessage.error('Failed') }
}

// Bindings
const bindings = ref<WorkStationBinding[]>([])
const selectedStationForBindings = ref('')
const loadBindings = async (code: string) => {
  selectedStationForBindings.value = code
  if (!code) { bindings.value = []; return }
  try { const res = await getStationBindingsAPI(code); bindings.value = (res.data as any) || [] } catch (_) { /* ignore */ }
}

const bindDialog = ref(false)
const bindForm = ref({ workstationCode: '', bindingType: 'EQUIPMENT', targetCode: '', targetName: '', targetType: '' })
const createBinding = async () => {
  try {
    const { workstationCode, bindingType, targetCode, targetName, targetType } = bindForm.value
    if (bindingType === 'EQUIPMENT') await bindEquipmentAPI(workstationCode, targetCode, targetName, targetType)
    else if (bindingType === 'TOOL') await bindToolAPI(workstationCode, targetCode, targetName, targetType)
    else await bindPersonnelAPI(workstationCode, targetCode, targetName, targetType)
    ElMessage.success('Bound')
    bindDialog.value = false
    loadBindings(workstationCode)
  } catch (_) { ElMessage.error('Failed') }
}

const unbind = async (id: number) => {
  try {
    await ElMessageBox.confirm('Unbind this item?', 'Confirm')
    await unbindStationAPI(id)
    ElMessage.success('Unbound')
    loadBindings(selectedStationForBindings.value)
  } catch (_) { /* ignore */ }
}

const refreshAll = () => { loadOrders(); loadStations() }

const statusTag = (s?: string): 'success' | 'warning' | 'danger' | 'info' => {
  const m: Record<string, any> = { PENDING: 'info', SCHEDULED: 'warning', IN_PROGRESS: 'success', PAUSED: 'warning', COMPLETED: 'success', CANCELLED: 'danger',
    IDLE: 'info', OPERATING: 'success', MAINTENANCE: 'warning', BREAKDOWN: 'danger', OFFLINE: 'danger' }
  return m[s || ''] || 'info'
}

onMounted(refreshAll)
</script>

<template>
  <div class="pm-container">
    <div class="toolbar">
      <el-button :icon="Refresh" @click="refreshAll">Refresh</el-button>
    </div>

    <el-tabs v-model="activeTab" type="border-card">
      <!-- Orders -->
      <el-tab-pane label="Production Orders" name="orders">
        <div class="section-toolbar">
          <el-button type="primary" :icon="Plus" @click="orderDialog = true">Create Order</el-button>
        </div>
        <el-table :data="orders" border stripe v-loading="loading">
          <el-table-column prop="orderCode" label="Code" width="130" />
          <el-table-column prop="productName" label="Product" min-width="150" />
          <el-table-column prop="quantityPlanned" label="Qty" width="60" />
          <el-table-column prop="priority" label="Priority" width="80">
            <template #default="{ row }"><el-tag :type="row.priority === 'URGENT' ? 'danger' : row.priority === 'HIGH' ? 'warning' : 'info'" size="small">{{ row.priority }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="progressPercentage" label="Progress" width="120">
            <template #default="{ row }">
              <el-progress :percentage="Math.round(row.progressPercentage || 0)" :status="row.status === 'COMPLETED' ? 'success' : undefined" />
            </template>
          </el-table-column>
          <el-table-column prop="status" label="Status" width="110">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="360" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.status === 'PENDING'" link type="primary" size="small" @click="openSchedule(row)">Schedule</el-button>
              <el-button v-if="row.status === 'SCHEDULED'" link type="success" size="small" @click="lifecycleAction(row.id, 'Start', startProductionAPI)">Start</el-button>
              <el-button v-if="row.status === 'IN_PROGRESS'" link type="warning" size="small" @click="lifecycleAction(row.id, 'Pause', pauseProductionAPI)">Pause</el-button>
              <el-button v-if="row.status === 'PAUSED'" link type="success" size="small" @click="lifecycleAction(row.id, 'Resume', resumeProductionAPI)">Resume</el-button>
              <el-button v-if="row.status === 'IN_PROGRESS'" link type="success" size="small" @click="lifecycleAction(row.id, 'Complete', completeProductionAPI)">Complete</el-button>
              <el-button v-if="!['COMPLETED', 'CANCELLED'].includes(row.status)" link type="danger" size="small" @click="lifecycleAction(row.id, 'Cancel', cancelOrderAPI)">Cancel</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Work Stations -->
      <el-tab-pane label="Work Stations" name="stations">
        <div class="section-toolbar">
          <el-button type="primary" :icon="Plus" @click="stationDialog = true">Create Station</el-button>
          <el-button @click="assignDialog = true">Assign Order</el-button>
        </div>
        <el-table :data="stations" border stripe v-loading="loading">
          <el-table-column prop="stationCode" label="Code" width="110" />
          <el-table-column prop="stationName" label="Name" min-width="140" />
          <el-table-column prop="stationType" label="Type" width="100" />
          <el-table-column prop="workshopCode" label="Workshop" width="100" />
          <el-table-column prop="capacity" label="Capacity" width="70" />
          <el-table-column prop="currentLoad" label="Load" width="60" />
          <el-table-column prop="currentOrderCode" label="Current Order" min-width="130" />
          <el-table-column prop="status" label="Status" width="110">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="120">
            <template #default="{ row }">
              <el-button link type="primary" size="small" @click="loadBindings(row.stationCode); activeTab = 'bindings'">Bindings</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Station Bindings -->
      <el-tab-pane label="Station Bindings" name="bindings">
        <div class="section-toolbar">
          <el-select v-model="selectedStationForBindings" placeholder="Select station" @change="loadBindings" style="width:200px; margin-right:8px">
            <el-option v-for="s in stations" :key="s.stationCode" :label="s.stationName" :value="s.stationCode" />
          </el-select>
          <el-button v-if="selectedStationForBindings" type="primary" :icon="Plus" @click="bindForm.workstationCode = selectedStationForBindings; bindDialog = true">Bind</el-button>
        </div>
        <el-table :data="bindings" border stripe>
          <el-table-column prop="bindingType" label="Type" width="100">
            <template #default="{ row }"><el-tag>{{ row.bindingType }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="targetCode" label="Code" width="120" />
          <el-table-column prop="targetName" label="Name" min-width="150" />
          <el-table-column prop="targetType" label="Target Type" width="110" />
          <el-table-column prop="isPrimary" label="Primary" width="70">
            <template #default="{ row }"><el-tag :type="row.isPrimary ? 'success' : 'info'" size="small">{{ row.isPrimary ? 'Yes' : 'No' }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="status" label="Status" width="80">
            <template #default="{ row }"><el-tag :type="row.status === 'ACTIVE' ? 'success' : 'info'" size="small">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="100">
            <template #default="{ row }">
              <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="unbind(row.id)">Unbind</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <!-- Dialogs -->
    <el-dialog v-model="orderDialog" title="Create Order" width="450px">
      <el-form :model="orderForm" label-width="120px">
        <el-form-item label="Product Code"><el-input v-model="orderForm.productCode" /></el-form-item>
        <el-form-item label="Product Name"><el-input v-model="orderForm.productName" /></el-form-item>
        <el-form-item label="Quantity"><el-input-number v-model="orderForm.quantity" :min="1" style="width:100%" /></el-form-item>
        <el-form-item label="Priority">
          <el-select v-model="orderForm.priority" style="width:100%">
            <el-option label="Low" value="LOW" /><el-option label="Normal" value="NORMAL" />
            <el-option label="High" value="HIGH" /><el-option label="Urgent" value="URGENT" />
          </el-select>
        </el-form-item>
        <el-form-item label="Workshop"><el-input v-model="orderForm.workshopCode" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="orderDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createOrder">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="stationDialog" title="Create Station" width="450px">
      <el-form :model="stationForm" label-width="120px">
        <el-form-item label="Station Code"><el-input v-model="stationForm.stationCode" /></el-form-item>
        <el-form-item label="Station Name"><el-input v-model="stationForm.stationName" /></el-form-item>
        <el-form-item label="Type">
          <el-select v-model="stationForm.stationType" style="width:100%">
            <el-option label="Assembly" value="ASSEMBLY" /><el-option label="Machining" value="MACHINING" />
            <el-option label="Inspection" value="INSPECTION" /><el-option label="Packaging" value="PACKAGING" />
          </el-select>
        </el-form-item>
        <el-form-item label="Workshop"><el-input v-model="stationForm.workshopCode" /></el-form-item>
        <el-form-item label="Capacity"><el-input-number v-model="stationForm.capacity" :min="1" style="width:100%" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="stationDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createStation">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="assignDialog" title="Assign Order to Station" width="400px">
      <el-form label-width="130px">
        <el-form-item label="Station Code"><el-input v-model="assignStationCode" /></el-form-item>
        <el-form-item label="Order Code"><el-input v-model="assignOrderCode" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="assignDialog = false">Cancel</el-button>
        <el-button type="primary" @click="assignStation">Assign</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="bindDialog" title="Create Binding" width="450px">
      <el-form :model="bindForm" label-width="130px">
        <el-form-item label="Binding Type">
          <el-select v-model="bindForm.bindingType" style="width:100%">
            <el-option label="Equipment" value="EQUIPMENT" /><el-option label="Tool" value="TOOL" />
            <el-option label="Personnel" value="PERSONNEL" />
          </el-select>
        </el-form-item>
        <el-form-item label="Target Code"><el-input v-model="bindForm.targetCode" /></el-form-item>
        <el-form-item label="Target Name"><el-input v-model="bindForm.targetName" /></el-form-item>
        <el-form-item label="Target Type"><el-input v-model="bindForm.targetType" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="bindDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createBinding">Bind</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.pm-container { padding: 20px; }
.toolbar { margin-bottom: 16px; }
.section-toolbar { margin-bottom: 12px; }
</style>
