<script setup lang=\"ts\">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Refresh, Document, Warning } from '@element-plus/icons-vue'
import {
  getTodayReconciliationStatusAPI,
  getReconciliationReportAPI,
  getPendingDifferenceCountAPI,
  type ReconciliationReport
} from '@/apis/payment'
import { formatDate, formatDateTime } from '@/utils/datetime'

const { t } = useI18n()

const loading = ref(false)
const reportData = ref<ReconciliationReport | null>(null)
const selectedDate = ref(formatDate(new Date()))
const pendingCount = ref(0)

const fetchReport = async () => {
  loading.value = true
  try {
    const response = await getReconciliationReportAPI(selectedDate.value)
    reportData.value = response.data
  } catch (error) {
    ElMessage.error(t('payment.reconciliation.report.loadFailed'))
  } finally {
    loading.value = false
  }
}

const fetchPendingCount = async () => {
  try {
    const response = await getPendingDifferenceCountAPI(selectedDate.value)
    pendingCount.value = response.data.count || 0
  } catch (error) {
    console.error('Failed to fetch pending count:', error)
  }
}

const fetchTodayStatus = async () => {
  loading.value = true
  try {
    const response = await getTodayReconciliationStatusAPI()
    reportData.value = response.data
  } catch (error) {
    ElMessage.error(t('payment.reconciliation.status.loadFailed'))
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchTodayStatus()
})

const handleDateChange = () => {
  fetchReport()
  fetchPendingCount()
}

const handleRefresh = () => {
  fetchTodayStatus()
  fetchPendingCount()
}

const getStatusColor = (status: string) => {
  if (!reportData.value) return 'info'
  if (reportData.value.failedCount > 0) return 'danger'
  if (reportData.value.pendingCount > 0) return 'warning'
  return 'success'
}
</script>

<template>
  <div class=\"app-container\">
    <el-row :gutter=\"20\">
      <el-col :span=\"24\">
        <el-card class=\"filter-card\" shadow=\"never\">
          <div class=\"filter-container\">
            <el-form :inline=\"true\">
              <el-form-item :label=\"t('payment.reconciliation.date.select')\">
                <el-date-picker
                  v-model=\"selectedDate\"
                  type=\"date\"
                  :placeholder=\"t('payment.reconciliation.date.placeholder')\"
                  format=\"YYYY-MM-DD\"
                  value-format=\"YYYY-MM-DD\"
                  @change=\"handleDateChange\"
                />
              </el-form-item>
              <el-form-item>
                <el-button type=\"primary\" :icon=\"Document\" @click=\"handleDateChange\">
                  {{ t('common.search') }}
                </el-button>
                <el-button :icon=\"Refresh\" @click=\"handleRefresh\">
                  {{ t('common.refresh') }}
                </el-button>
              </el-form-item>
            </el-form>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter=\"20\" style=\"margin-top: 20px\">
      <el-col :span=\"6\">
        <el-card shadow=\"never\">
          <div class=\"stat-card\">
            <div class=\"stat-icon success\">
              <el-icon size=\"30\"><Document /></el-icon>
            </div>
            <div class=\"stat-content\">
              <div class=\"stat-label\">{{ t('payment.reconciliation.stat.totalOrders') }}</div>
              <div class=\"stat-value\">{{ reportData?.totalOrders || 0 }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span=\"6\">
        <el-card shadow=\"never\">
          <div class=\"stat-card\">
            <div class=\"stat-icon primary\">
              <el-icon size=\"30\"><Document /></el-icon>
            </div>
            <div class=\"stat-content\">
              <div class=\"stat-label\">{{ t('payment.reconciliation.stat.totalAmount') }}</div>
              <div class=\"stat-value\">¥{{ (reportData?.totalAmount || 0).toFixed(2) }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span=\"6\">
        <el-card shadow=\"never\">
          <div class=\"stat-card\">
            <div class=\"stat-icon success\">
              <el-icon size=\"30\"><Document /></el-icon>
            </div>
            <div class=\"stat-content\">
              <div class=\"stat-label\">{{ t('payment.reconciliation.stat.successCount') }}</div>
              <div class=\"stat-value\">{{ reportData?.successCount || 0 }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span=\"6\">
        <el-card shadow=\"never\">
          <div class=\"stat-card\">
            <div class=\"stat-icon danger\">
              <el-icon size=\"30\"><Document /></el-icon>
            </div>
            <div class=\"stat-content\">
              <div class=\"stat-label\">{{ t('payment.reconciliation.stat.failedCount') }}</div>
              <div class=\"stat-value\">{{ reportData?.failedCount || 0 }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter=\"20\" style=\"margin-top: 20px\">
      <el-col :span=\"6\">
        <el-card shadow=\"never\">
          <div class=\"stat-card\">
            <div class=\"stat-icon warning\">
              <el-icon size=\"30\"><Document /></el-icon>
            </div>
            <div class=\"stat-content\">
              <div class=\"stat-label\">{{ t('payment.reconciliation.stat.pendingCount') }}</div>
              <div class=\"stat-value\">{{ reportData?.pendingCount || 0 }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span=\"6\">
        <el-card shadow=\"never\">
          <div class=\"stat-card\">
            <div class=\"stat-icon\" :class=\"pendingCount > 0 ? 'danger' : 'info'\">
              <el-icon size=\"30\"><Warning /></el-icon>
            </div>
            <div class=\"stat-content\">
              <div class=\"stat-label\">{{ t('payment.reconciliation.stat.differenceCount') }}</div>
              <div class=\"stat-value\">{{ pendingCount }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span=\"12\">
        <el-card shadow=\"never\" class=\"status-card\">
          <template #header>
            <span>{{ t('payment.reconciliation.status.title') }}</span>
          </template>
          <div class=\"status-content\">
            <el-result
              icon=\"success\"
              :title=\"t('payment.reconciliation.status.normal')\"
              :sub-title=\"t('payment.reconciliation.status.normalDesc')\"
            >
              <template #extra>
                <el-tag :type=\"getStatusColor('status')\" size=\"large\">
                  {{ t('payment.reconciliation.status.lastUpdate') }}: {{ formatDateTime(reportData?.date) }}
                </el-tag>
              </template>
            </el-result>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter=\"20\" style=\"margin-top: 20px\">
      <el-col :span=\"24\">
        <el-card shadow=\"never\">
          <template #header>
            <span>{{ t('payment.reconciliation.chart.title') }}</span>
          </template>
          <div class=\"chart-container\">
            <el-empty :description=\"t('payment.reconciliation.chart.empty')\" />
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.filter-card {
  margin-bottom: 20px;
}

.filter-container {
  display: flex;
  align-items: center;
}

.stat-card {
  display: flex;
  align-items: center;
  padding: 10px;
}

.stat-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 60px;
  height: 60px;
  border-radius: 10px;
  margin-right: 15px;
}

.stat-icon.success {
  background-color: #f0f9eb;
  color: #67c23a;
}

.stat-icon.primary {
  background-color: #ecf5ff;
  color: #409eff;
}

.stat-icon.warning {
  background-color: #fdf6ec;
  color: #e6a23c;
}

.stat-icon.danger {
  background-color: #fef0f0;
  color: #f56c6c;
}

.stat-icon.info {
  background-color: #f4f4f5;
  color: #909399;
}

.stat-content {
  flex: 1;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 5px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.status-card {
  height: 100%;
}

.status-content {
  display: flex;
  justify-content: center;
  align-items: center;
}

.chart-container {
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>