<template>
  <div class="inventory-alert-container">
    <el-card class="filter-card">
      <el-form :inline="true" :model="queryParams" class="demo-form-inline">
        <el-form-item label="Alert Status">
          <el-select v-model="queryParams.status" placeholder="Select Status" clearable>
            <el-option label="Pending" :value="0" />
            <el-option label="Notified" :value="1" />
            <el-option label="Resolved" :value="2" />
            <el-option label="Ignored" :value="3" />
          </el-select>
        </el-form-item>
        <el-form-item label="Alert Level">
          <el-select v-model="queryParams.alertLevel" placeholder="Select Level" clearable>
            <el-option label="Low" :value="1" />
            <el-option label="Medium" :value="2" />
            <el-option label="High" :value="3" />
            <el-option label="Critical" :value="4" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">Query</el-button>
          <el-button @click="resetQuery">Reset</el-button>
          <el-button type="warning" @click="loadHighPriority">High Priority</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card>
      <div class="stats-cards">
        <el-row :gutter="20">
          <el-col :span="6">
            <div class="stat-card pending">
              <div class="stat-value">{{ stats.pending }}</div>
              <div class="stat-label">Pending</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="stat-card notified">
              <div class="stat-value">{{ stats.notified }}</div>
              <div class="stat-label">Notified</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="stat-card resolved">
              <div class="stat-value">{{ stats.resolved }}</div>
              <div class="stat-label">Resolved</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="stat-card critical">
              <div class="stat-value">{{ stats.critical }}</div>
              <div class="stat-label">Critical</div>
            </div>
          </el-col>
        </el-row>
      </div>

      <el-table :data="alertList" border style="width: 100%; margin-top: 20px">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="productName" label="Product Name" width="180" />
        <el-table-column prop="skuCode" label="SKU Code" width="120" />
        <el-table-column prop="warehouseName" label="Warehouse" width="120" />
        <el-table-column label="Stock Info" width="150">
          <template #default="{ row }">
            <span :class="{ 'stock-warning': row.currentStock < row.threshold }">
              {{ row.currentStock }} / {{ row.threshold }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="alertLevelDesc" label="Level" width="100">
          <template #default="{ row }">
            <el-tag :type="getLevelTagType(row.alertLevel)">
              {{ row.alertLevelDesc }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="alertStatusDesc" label="Status" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusTagType(row.alertStatus)">
              {{ row.alertStatusDesc }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="alertMessage" label="Message" min-width="200" />
        <el-table-column prop="alertTime" label="Alert Time" width="180" />
        <el-table-column label="Operations" fixed="right" width="200">
          <template #default="{ row }">
            <el-button
              v-if="row.alertStatus === 0 || row.alertStatus === 1"
              type="success"
              link
              @click="handleResolve(row)"
            >
              Resolve
            </el-button>
            <el-button
              v-if="row.alertStatus === 0 || row.alertStatus === 1"
              type="info"
              link
              @click="handleIgnore(row)"
            >
              Ignore
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- Resolve Dialog -->
    <el-dialog v-model="resolveDialogVisible" title="Resolve Alert" width="400px">
      <el-form ref="resolveFormRef" :model="resolveForm" label-width="100px">
        <el-form-item label="Alert ID">
          <el-input :value="resolveForm.alertId" disabled />
        </el-form-item>
        <el-form-item label="Remark">
          <el-input v-model="resolveForm.remark" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="resolveDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="submitResolve">Confirm</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance } from 'element-plus'
import {
  getAlertListAPI,
  getAlertsByStatusAPI,
  getAlertsByLevelAPI,
  getHighPriorityAlertsAPI,
  resolveAlertAPI,
  ignoreAlertAPI,
  type InventoryAlert
} from '@/apis/inventoryAlert'

const queryParams = reactive({
  status: null as number | null,
  alertLevel: null as number | null
})

const alertList = ref<InventoryAlert[]>([])
const resolveDialogVisible = ref(false)
const resolveFormRef = ref<FormInstance>()

const resolveForm = reactive({
  alertId: 0,
  remark: ''
})

const stats = reactive({
  pending: 0,
  notified: 0,
  resolved: 0,
  critical: 0
})

const getLevelTagType = (level: number): 'success' | 'warning' | 'danger' | 'info' => {
  const levelMap: Record<number, 'success' | 'warning' | 'danger' | 'info'> = {
    1: 'info',
    2: 'warning',
    3: 'danger',
    4: 'danger'
  }
  return levelMap[level] || 'info'
}

const getStatusTagType = (status: number): 'success' | 'warning' | 'danger' | 'info' => {
  const statusMap: Record<number, 'success' | 'warning' | 'danger' | 'info'> = {
    0: 'warning',
    1: 'danger',
    2: 'success',
    3: 'info'
  }
  return statusMap[status] || 'info'
}

const updateStats = (list: InventoryAlert[]) => {
  stats.pending = list.filter(a => a.alertStatus === 0).length
  stats.notified = list.filter(a => a.alertStatus === 1).length
  stats.resolved = list.filter(a => a.alertStatus === 2).length
  stats.critical = list.filter(a => a.alertLevel === 4 && (a.alertStatus === 0 || a.alertStatus === 1)).length
}

const loadAlertList = async () => {
  try {
    const response = await getAlertListAPI()
    alertList.value = response.data || []
    updateStats(alertList.value)
  } catch (error) {
    ElMessage.error('Failed to load alerts')
  }
}

const handleQuery = async () => {
  try {
    let response
    if (queryParams.status !== null) {
      response = await getAlertsByStatusAPI(queryParams.status)
    } else if (queryParams.alertLevel !== null) {
      response = await getAlertsByLevelAPI(queryParams.alertLevel)
    } else {
      response = await getAlertListAPI()
    }
    alertList.value = response.data || []
    updateStats(alertList.value)
  } catch (error) {
    ElMessage.error('Failed to load alerts')
  }
}

const resetQuery = () => {
  queryParams.status = null
  queryParams.alertLevel = null
  loadAlertList()
}

const loadHighPriority = async () => {
  try {
    const response = await getHighPriorityAlertsAPI()
    alertList.value = response.data || []
    updateStats(alertList.value)
  } catch (error) {
    ElMessage.error('Failed to load high priority alerts')
  }
}

const handleResolve = (row: InventoryAlert) => {
  resolveForm.alertId = row.id
  resolveForm.remark = ''
  resolveDialogVisible.value = true
}

const submitResolve = async () => {
  try {
    await resolveAlertAPI(resolveForm.alertId, resolveForm.remark)
    ElMessage.success('Alert resolved successfully')
    resolveDialogVisible.value = false
    loadAlertList()
  } catch (error) {
    ElMessage.error('Failed to resolve alert')
  }
}

const handleIgnore = async (row: InventoryAlert) => {
  try {
    await ElMessageBox.confirm('Are you sure to ignore this alert?', 'Warning', {
      confirmButtonText: 'Confirm',
      cancelButtonText: 'Cancel',
      type: 'warning'
    })
    await ignoreAlertAPI(row.id)
    ElMessage.success('Alert ignored')
    loadAlertList()
  } catch {
    // User cancelled
  }
}

onMounted(() => {
  loadAlertList()
})
</script>

<style scoped>
.inventory-alert-container {
  padding: 20px;
}

.filter-card {
  margin-bottom: 20px;
}

.stats-cards {
  margin-bottom: 10px;
}

.stat-card {
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  color: #fff;
}

.stat-card.pending {
  background: linear-gradient(135deg, #f39c12, #f1c40f);
}

.stat-card.notified {
  background: linear-gradient(135deg, #e74c3c, #c0392b);
}

.stat-card.resolved {
  background: linear-gradient(135deg, #27ae60, #2ecc71);
}

.stat-card.critical {
  background: linear-gradient(135deg, #8e44ad, #9b59b6);
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
}

.stat-label {
  font-size: 14px;
  margin-top: 5px;
}

.stock-warning {
  color: #e74c3c;
  font-weight: bold;
}
</style>