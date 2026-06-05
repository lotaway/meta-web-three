<template>
  <div class="app-container">
    <div v-if="loading" v-loading="loading" style="height: 200px" />
    <div v-else-if="order">
      <el-card class="box-card">
        <template #header>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>{{ t('mes.scheduling.detail') }} - {{ order.scheduleNo }}</span>
            <div>
              <el-button v-if="order.status === 'SCHEDULED'" type="success" @click="handleStart">
                {{ t('mes.scheduling.start') }}
              </el-button>
              <el-button v-if="order.status === 'IN_PROGRESS'" type="success" @click="handleComplete">
                {{ t('mes.scheduling.complete') }}
              </el-button>
              <el-button v-if="order.status !== 'COMPLETED' && order.status !== 'CANCELLED'"
                type="danger" @click="handleCancel">
                {{ t('mes.scheduling.cancel') }}
              </el-button>
              <el-button @click="handleBack">{{ t('common.back') }}</el-button>
            </div>
          </div>
        </template>
        <el-descriptions :column="3" border>
          <el-descriptions-item :label="t('mes.scheduling.scheduleNo')" :span="1">
            {{ order.scheduleNo }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.orderNo')" :span="1">
            {{ order.orderNo }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.status')" :span="1">
            <el-tag :type="getStatusType(order.status)">
              {{ t(`mes.scheduling.status${order.status}`) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.productCode')">
            {{ order.productCode }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.productName')">
            {{ order.productName }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.priority')">
            <el-tag :type="getPriorityType(order.priority)" size="small">
              {{ order.priority }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.quantity')">
            {{ order.quantity }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.completedQty')">
            {{ order.completedQuantity || 0 }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.workshop')">
            {{ order.workshopId }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.dueDate')">
            {{ order.dueDate ? formatDateTime(order.dueDate) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.scheduledStart')">
            {{ order.scheduledStartTime ? formatDateTime(order.scheduledStartTime) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.scheduledEnd')">
            {{ order.scheduledEndTime ? formatDateTime(order.scheduledEndTime) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.routeCode')">
            {{ order.routeCode || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.createdBy')">
            {{ order.createdBy || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.scheduling.createdAt')">
            {{ order.createdAt ? formatDateTime(order.createdAt) : '-' }}
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <el-card class="box-card" style="margin-top: 16px;">
        <template #header>
          <span>{{ t('mes.scheduling.operations') }}</span>
        </template>
        <el-table v-if="order.operations && order.operations.length > 0" :data="order.operations" border>
          <el-table-column label="#" prop="sequenceNo" width="60" />
          <el-table-column :label="t('mes.scheduling.operationCode')" prop="operationCode" width="120" />
          <el-table-column :label="t('mes.scheduling.operationName')" prop="operationName" min-width="120" />
          <el-table-column :label="t('mes.scheduling.resourceCode')" prop="resourceCode" width="120" />
          <el-table-column :label="t('mes.scheduling.resourceName')" prop="resourceName" width="120" />
          <el-table-column :label="t('mes.scheduling.setupTime')" prop="setupTimeMinutes" width="80" />
          <el-table-column :label="t('mes.scheduling.processingTime')" prop="processingTimeMinutes" width="80" />
          <el-table-column :label="t('mes.scheduling.teardownTime')" prop="teardownTimeMinutes" width="80" />
          <el-table-column :label="t('mes.scheduling.opStatus')" prop="status" width="100">
            <template #default="{ row }">
              <el-tag :type="getOpStatusType(row.status)">
                {{ t(`mes.scheduling.opStatus${row.status}`) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.scheduledStart')" width="140">
            <template #default="{ row }">
              {{ row.scheduledStartTime ? formatDateTime(row.scheduledStartTime) : '-' }}
            </template>
          </el-table-column>
          <el-table-column :label="t('mes.scheduling.scheduledEnd')" width="140">
            <template #default="{ row }">
              {{ row.scheduledEndTime ? formatDateTime(row.scheduledEndTime) : '-' }}
            </template>
          </el-table-column>
        </el-table>
        <el-empty v-else :description="t('mes.scheduling.noOperations')" />
      </el-card>
    </div>
    <div v-else>
      <el-empty :description="t('mes.scheduling.dataNotExist')" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import {
  getScheduleOrderByIdAPI,
  startOrderAPI,
  completeOrderAPI,
  cancelOrderAPI,
  type ScheduleOrder,
} from '@/apis/scheduling'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const order = ref<ScheduleOrder | null>(null)

function getStatusType(status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = {
    PENDING: 'info', SCHEDULED: 'primary', IN_PROGRESS: 'warning',
    COMPLETED: 'success', DELAYED: 'danger', CANCELLED: 'info',
  }
  return map[status] || 'info'
}

function getPriorityType(p: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = {
    LOW: 'info', NORMAL: 'primary', HIGH: 'warning', URGENT: 'danger',
  }
  return map[p] || 'info'
}

function getOpStatusType(s: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' {
  const map: Record<string, any> = {
    PENDING: 'info', SCHEDULED: 'primary', IN_PROGRESS: 'warning',
    COMPLETED: 'success', BLOCKED: 'danger',
  }
  return map[s] || 'info'
}

function formatDateTime(dateStr: string) {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

async function loadOrder(id: number) {
  loading.value = true
  try {
    const res = await getScheduleOrderByIdAPI(id)
    order.value = res.data
  } catch {
    ElMessage.error(t('mes.scheduling.loadFailed'))
  } finally {
    loading.value = false
  }
}

async function handleStart() {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmStart'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'info' })
    await startOrderAPI(order.value!.id!)
    ElMessage.success(t('mes.scheduling.startSuccess'))
    loadOrder(order.value!.id!)
  } catch { /* cancel */ }
}

async function handleComplete() {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmComplete'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'info' })
    await completeOrderAPI(order.value!.id!)
    ElMessage.success(t('mes.scheduling.completeSuccess'))
    loadOrder(order.value!.id!)
  } catch { /* cancel */ }
}

async function handleCancel() {
  try {
    await ElMessageBox.confirm(t('mes.scheduling.confirmCancel'), t('common.warning'),
      { confirmButtonText: t('common.confirm'), cancelButtonText: t('common.cancel'), type: 'warning' })
    await cancelOrderAPI(order.value!.id!)
    ElMessage.success(t('mes.scheduling.cancelSuccess'))
    loadOrder(order.value!.id!)
  } catch { /* cancel */ }
}

function handleBack() {
  router.push({ name: 'scheduling' })
}

onMounted(() => {
  if (route.query.id) {
    loadOrder(Number(route.query.id))
  }
})
</script>
