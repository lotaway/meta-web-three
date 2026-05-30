<template>
  <div class="app-container">
    <el-card class="detail-card">
      <template #header>
        <div class="card-header">
          <span>{{ t('mes.workOrder.detail') }}</span>
          <el-button type="primary" @click="handleBack">
            {{ t('mes.workOrder.back') }}
          </el-button>
        </div>
      </template>

      <div v-loading="loading" class="detail-content">
        <el-descriptions :column="2" border>
          <el-descriptions-item :label="t('common.id')">
            {{ workOrder?.id }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.workOrderNo')">
            {{ workOrder?.workOrderNo }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.productCode')">
            {{ workOrder?.productCode }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.productName')">
            {{ workOrder?.productName }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.quantity')">
            {{ workOrder?.quantity }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.completedQuantity')">
            {{ workOrder?.completedQuantity || 0 }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.status')">
            <el-tag :type="getStatusType(workOrder?.status || '')">
              {{ workOrder?.status ? t(`mes.workOrder.status${workOrder.status}`) : '-' }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.typeCode')">
            {{ workOrder?.typeCode ? t(`mes.workOrder.typeCode${workOrder.typeCode}`) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.priority')">
            {{ workOrder?.priority ? t(`mes.workOrder.priority${workOrder.priority}`) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.completionRate')">
            {{ workOrder?.completionRate ? `${workOrder.completionRate}%` : '0%' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.workshopId')">
            {{ workOrder?.workshopId || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.processRouteId')">
            {{ workOrder?.processRouteId || '-' }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.plannedStartTime')">
            {{ formatDateTime(workOrder?.plannedStartTime) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.plannedEndTime')">
            {{ formatDateTime(workOrder?.plannedEndTime) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.actualStartTime')">
            {{ formatDateTime(workOrder?.actualStartTime) }}
          </el-descriptions-item>
          <el-descriptions-item :label="t('mes.workOrder.actualEndTime')">
            {{ formatDateTime(workOrder?.actualEndTime) }}
          </el-descriptions-item>
        </el-descriptions>

        <!-- 操作按钮 -->
        <div class="action-buttons">
          <el-button
            v-if="canRelease"
            type="primary"
            @click="handleRelease"
          >
            {{ t('mes.workOrder.release') }}
          </el-button>
          <el-button
            v-if="canStart"
            type="success"
            @click="handleStart"
          >
            {{ t('mes.workOrder.start') }}
          </el-button>
          <el-button
            v-if="canPause"
            type="warning"
            @click="handlePause"
          >
            {{ t('mes.workOrder.pause') }}
          </el-button>
          <el-button
            v-if="canResume"
            type="success"
            @click="handleResume"
          >
            {{ t('mes.workOrder.resume') }}
          </el-button>
          <el-button
            v-if="canComplete"
            type="success"
            @click="handleComplete"
          >
            {{ t('mes.workOrder.complete') }}
          </el-button>
          <el-button
            v-if="canCancel"
            type="danger"
            @click="handleCancel"
          >
            {{ t('mes.workOrder.cancelAction') }}
          </el-button>
          <el-button
            v-if="canSplit"
            type="info"
            @click="handleSplit"
          >
            {{ t('mes.workOrder.split') }}
          </el-button>
          <el-button
            v-if="canUpdate"
            type="primary"
            @click="handleUpdate"
          >
            {{ t('common.edit') }}
          </el-button>
        </div>

        <!-- 子工单列表 -->
        <div v-if="childOrders.length > 0" class="child-orders">
          <h3>{{ t('mes.workOrder.childOrders') }}</h3>
          <el-table :data="childOrders" border style="width: 100%">
            <el-table-column :label="t('common.id')" prop="id" width="80" />
            <el-table-column :label="t('mes.workOrder.workOrderNo')" prop="workOrderNo" min-width="120" />
            <el-table-column :label="t('mes.workOrder.quantity')" prop="quantity" width="100" />
            <el-table-column :label="t('mes.workOrder.status')" prop="status" width="120">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)">
                  {{ t(`mes.workOrder.status${row.status}`) }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-card>

    <!-- 拆分对话框 -->
    <el-dialog v-model="splitDialogVisible" :title="t('mes.workOrder.split')" width="400px">
      <el-form :model="splitForm" label-width="100px">
        <el-form-item :label="t('mes.workOrder.splitType')">
          <el-select v-model="splitForm.splitType">
            <el-option value="BY_BOM" :label="t('mes.workOrder.splitTypeBY_BOM')" />
            <el-option value="BY_PROCESS" :label="t('mes.workOrder.splitTypeBY_PROCESS')" />
            <el-option value="MANUAL" :label="t('mes.workOrder.splitTypeMANUAL')" />
          </el-select>
        </el-form-item>
        <el-form-item :label="t('mes.workOrder.splitCount')">
          <el-input-number v-model="splitForm.splitCount" :min="2" :max="10" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="splitDialogVisible = false">{{ t('mes.workOrder.cancel') }}</el-button>
        <el-button type="primary" @click="confirmSplit">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>

    <!-- 取消原因对话框 -->
    <el-dialog v-model="cancelDialogVisible" :title="t('mes.workOrder.cancelAction')" width="400px">
      <el-form :model="cancelForm" label-width="100px">
        <el-form-item :label="t('mes.workOrder.cancelReason')">
          <el-input
            v-model="cancelForm.reason"
            type="textarea"
            :placeholder="t('mes.workOrder.cancelReasonPlaceholder')"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="cancelDialogVisible = false">{{ t('mes.workOrder.cancel') }}</el-button>
        <el-button type="primary" @click="confirmCancel">{{ t('common.confirm') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import {
  getWorkOrderByIdAPI,
  releaseWorkOrderAPI,
  startWorkOrderAPI,
  pauseWorkOrderAPI,
  resumeWorkOrderAPI,
  completeWorkOrderAPI,
  cancelWorkOrderAPI,
  splitWorkOrderAPI,
  getChildWorkOrdersAPI,
  type WorkOrder,
  type SplitType
} from '@/apis/workOrder'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const workOrder = ref<WorkOrder | null>(null)
const childOrders = ref<WorkOrder[]>([])

// 对话框
const splitDialogVisible = ref(false)
const splitForm = ref({
  splitType: 'BY_BOM' as SplitType,
  splitCount: 2,
})
const cancelDialogVisible = ref(false)
const cancelForm = ref({
  reason: '',
})

// 操作权限
const canRelease = computed(() => workOrder.value?.status === 'DRAFT')
const canStart = computed(() => workOrder.value?.status === 'RELEASED')
const canPause = computed(() => workOrder.value?.status === 'IN_PROGRESS')
const canResume = computed(() => workOrder.value?.status === 'PAUSED')
const canComplete = computed(() => ['RELEASED', 'IN_PROGRESS'].includes(workOrder.value?.status || ''))
const canCancel = computed(() => ['DRAFT', 'RELEASED', 'IN_PROGRESS', 'PAUSED'].includes(workOrder.value?.status || ''))
const canSplit = computed(() => workOrder.value?.status === 'COMPLETED')
const canUpdate = computed(() => ['DRAFT'].includes(workOrder.value?.status || ''))

const getStatusType = (status: string): 'primary' | 'success' | 'warning' | 'info' | 'danger' => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'info' | 'danger'> = {
    DRAFT: 'info',
    RELEASED: 'warning',
    IN_PROGRESS: 'primary',
    PAUSED: 'warning',
    COMPLETED: 'success',
    CANCELLED: 'info',
  }
  return typeMap[status] || 'info'
}

const formatDateTime = (dateStr?: string) => {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleString()
}

const loadData = async () => {
  const id = Number(route.query.id)
  if (!id) {
    handleBack()
    return
  }

  loading.value = true
  try {
    const response = await getWorkOrderByIdAPI(id)
    workOrder.value = response.data || null

    // 加载子工单
    if (workOrder.value?.id) {
      const childResponse = await getChildWorkOrdersAPI(workOrder.value.id)
      childOrders.value = childResponse.data || []
    }
  } catch (error) {
    ElMessage.error(t('mes.workOrder.loadFailed'))
  } finally {
    loading.value = false
  }
}

const handleRelease = async () => {
  try {
    await releaseWorkOrderAPI(workOrder.value!.id!)
    ElMessage.success(t('mes.workOrder.releaseSuccess'))
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.releaseFailed'))
  }
}

const handleStart = async () => {
  try {
    await startWorkOrderAPI(workOrder.value!.id!)
    ElMessage.success(t('mes.workOrder.startSuccess'))
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.startFailed'))
  }
}

const handlePause = async () => {
  try {
    await pauseWorkOrderAPI(workOrder.value!.id!)
    ElMessage.success(t('mes.workOrder.pauseSuccess'))
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.pauseFailed'))
  }
}

const handleResume = async () => {
  try {
    await resumeWorkOrderAPI(workOrder.value!.id!)
    ElMessage.success(t('mes.workOrder.resumeSuccess'))
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.resumeFailed'))
  }
}

const handleComplete = async () => {
  try {
    await completeWorkOrderAPI(workOrder.value!.id!)
    ElMessage.success(t('mes.workOrder.completeSuccess'))
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.completeFailed'))
  }
}

const handleCancel = () => {
  cancelForm.value.reason = ''
  cancelDialogVisible.value = true
}

const confirmCancel = async () => {
  try {
    await cancelWorkOrderAPI(workOrder.value!.id!, { reason: cancelForm.value.reason })
    ElMessage.success(t('mes.workOrder.cancelSuccess'))
    cancelDialogVisible.value = false
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.cancelFailed'))
  }
}

const handleSplit = () => {
  splitForm.value = {
    splitType: 'BY_BOM',
    splitCount: 2,
  }
  splitDialogVisible.value = true
}

const confirmSplit = async () => {
  try {
    await splitWorkOrderAPI(workOrder.value!.id!, {
      splitType: splitForm.value.splitType,
      splitCount: splitForm.value.splitCount,
    })
    ElMessage.success(t('mes.workOrder.splitSuccess'))
    splitDialogVisible.value = false
    loadData()
  } catch {
    ElMessage.error(t('mes.workOrder.splitFailed'))
  }
}

const handleUpdate = () => {
  router.push({ name: 'workOrderForm', query: { id: workOrder.value?.id } })
}

const handleBack = () => {
  router.push({ name: 'workOrder' })
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.detail-card {
  max-width: 1200px;
  margin: 0 auto;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: bold;
}
.detail-content {
  padding: 20px;
}
.action-buttons {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
}
.action-buttons .el-button {
  margin-right: 10px;
  margin-bottom: 10px;
}
.child-orders {
  margin-top: 30px;
}
.child-orders h3 {
  margin-bottom: 15px;
  font-size: 16px;
  color: #303133;
}
</style>