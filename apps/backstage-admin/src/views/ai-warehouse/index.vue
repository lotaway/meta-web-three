<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import {
  listAICapabilitiesAPI, enableAICapabilityAPI,
  listAIWarehouseRequestsAPI,
  getLocationRecommendationAPI, getDemandForecastAPI,
  getRestockSuggestionAPI, detectAnomaliesAPI,
  type AICapability, type AIWarehouseRequest,
} from '@/apis/ai-warehouse'

const activeTab = ref('dashboard')
const loading = ref(false)

// Capabilities
const capabilities = ref<AICapability[]>([])
const loadCapabilities = async () => {
  try { const res = await listAICapabilitiesAPI(); capabilities.value = (res.data as any) || [] } catch (e: any) { console.error('[loadCapabilities]', e); ElMessage.error('加载失败') }
}

const toggleCap = async (cap: AICapability) => {
  try {
    await enableAICapabilityAPI((cap as any).capabilityId || cap.id, !cap.enabled)
    cap.enabled = !cap.enabled
    ElMessage.success(cap.enabled ? 'Enabled' : 'Disabled')
  } catch (_) { ElMessage.error('Failed to toggle') }
}

// Requests
const requests = ref<AIWarehouseRequest[]>([])
const loadRequests = async () => {
  try {
    const res = await listAIWarehouseRequestsAPI({ pageNum: 1, pageSize: 50 })
    requests.value = (res.data as any)?.list || []
  } catch (e: any) { console.error('[loadRequests]', e); ElMessage.error('加载失败') }
}

// Location Recommendation
const locResult = ref<any[]>([])
const loadLocRec = async () => {
  try { const res = await getLocationRecommendationAPI(1, {}); locResult.value = (res.data as any)?.recommendations || [] } catch (e: any) { console.error('[loadLocRec]', e); ElMessage.error('加载失败') }
}

// Demand Forecast
const forecastResult = ref<any[]>([])
const loadForecast = async () => {
  try { const res = await getDemandForecastAPI(1, {}); forecastResult.value = (res.data as any)?.forecasts || [] } catch (e: any) { console.error('[loadForecast]', e); ElMessage.error('加载失败') }
}

// Restock Suggestion
const restockResult = ref<any[]>([])
const loadRestock = async () => {
  try { const res = await getRestockSuggestionAPI(1, {}); restockResult.value = (res.data as any)?.suggestions || [] } catch (e: any) { console.error('[loadRestock]', e); ElMessage.error('加载失败') }
}

// Anomaly Detection
const anomalyResult = ref<any[]>([])
const loadAnomalies = async () => {
  try { const res = await detectAnomaliesAPI(1, {}); anomalyResult.value = (res.data as any)?.anomalies || [] } catch (e: any) { console.error('[loadAnomalies]', e); ElMessage.error('加载失败') }
}

const refreshAll = () => {
  loadCapabilities(); loadRequests(); loadLocRec(); loadForecast(); loadRestock(); loadAnomalies()
}

const statusTag = (s?: string) => {
  const m: Record<string, string> = { PENDING: 'info', PROCESSING: 'warning', SUCCESS: 'success', FALLBACK_USED: 'warning', FAILED: 'danger' }
  return m[s || ''] || 'info'
}

onMounted(refreshAll)
</script>

<template>
  <div class="aw-container">
    <div class="toolbar">
      <el-button :icon="Refresh" @click="refreshAll">Refresh</el-button>
    </div>

    <el-tabs v-model="activeTab" type="border-card">
      <!-- Dashboard -->
      <el-tab-pane label="Dashboard" name="dashboard">
        <el-row :gutter="20">
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ capabilities.length }}</div><div class="stat-label">Capabilities</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ capabilities.filter(c => c.enabled).length }}</div><div class="stat-label">Enabled</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ requests.length }}</div><div class="stat-label">Total Requests</div></el-card></el-col>
          <el-col :span="6"><el-card shadow="hover"><div class="stat-value">{{ requests.filter(r => r.status === 'SUCCESS').length }}</div><div class="stat-label">Successful</div></el-card></el-col>
        </el-row>
      </el-tab-pane>

      <!-- Capabilities -->
      <el-tab-pane label="AI Capabilities" name="capabilities">
        <el-table :data="capabilities" border stripe v-loading="loading">
          <el-table-column label="ID" width="180">
            <template #default="{ row }">{{ row.capabilityId || row.id }}</template>
          </el-table-column>
          <el-table-column prop="capabilityName" label="Name" min-width="160" />
          <el-table-column prop="type" label="Type" width="150" />
          <el-table-column prop="endpoint" label="Endpoint" min-width="200" />
          <el-table-column prop="enabled" label="Status" width="90">
            <template #default="{ row }"><el-tag :type="row.enabled ? 'success' : 'info'">{{ row.enabled ? 'Enabled' : 'Disabled' }}</el-tag></template>
          </el-table-column>
          <el-table-column label="Actions" width="120" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.enabled" link type="warning" size="small" @click="toggleCap(row)">Disable</el-button>
              <el-button v-else link type="success" size="small" @click="toggleCap(row)">Enable</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Request Records -->
      <el-tab-pane label="Request Records" name="requests">
        <el-table :data="requests" border stripe v-loading="loading">
          <el-table-column prop="id" label="ID" width="60" />
          <el-table-column prop="warehouseName" label="Warehouse" width="120" />
          <el-table-column prop="capabilityType" label="Capability" width="160" />
          <el-table-column prop="requestData" label="Request" min-width="200" show-overflow-tooltip />
          <el-table-column prop="responseData" label="Response" min-width="200" show-overflow-tooltip />
          <el-table-column prop="status" label="Status" width="120">
            <template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="createdAt" label="Created" width="170" />
        </el-table>
      </el-tab-pane>

      <!-- Location Recommendation -->
      <el-tab-pane label="Location Rec" name="location">
        <div class="section-toolbar">
          <el-button type="primary" size="small" @click="loadLocRec">Query</el-button>
        </div>
        <el-table :data="locResult" border stripe>
          <el-table-column prop="locationId" label="Location ID" width="100" />
          <el-table-column prop="score" label="Score" width="100">
            <template #default="{ row }">{{ (row.score * 100).toFixed(0) }}%</template>
          </el-table-column>
          <el-table-column prop="reason" label="Reason" min-width="300" />
        </el-table>
      </el-tab-pane>

      <!-- Demand Forecast -->
      <el-tab-pane label="Demand Forecast" name="forecast">
        <div class="section-toolbar">
          <el-button type="primary" size="small" @click="loadForecast">Query</el-button>
        </div>
        <el-table :data="forecastResult" border stripe>
          <el-table-column prop="date" label="Date" width="120" />
          <el-table-column prop="quantity" label="Quantity" width="100" />
          <el-table-column prop="confidence" label="Confidence" width="100">
            <template #default="{ row }">{{ (row.confidence * 100).toFixed(0) }}%</template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Restock Suggestion -->
      <el-tab-pane label="Restock Suggestion" name="restock">
        <div class="section-toolbar">
          <el-button type="primary" size="small" @click="loadRestock">Query</el-button>
        </div>
        <el-table :data="restockResult" border stripe>
          <el-table-column prop="skuCode" label="SKU" width="120" />
          <el-table-column prop="quantity" label="Qty" width="80" />
          <el-table-column prop="urgency" label="Urgency" width="100">
            <template #default="{ row }"><el-tag :type="row.urgency === 'HIGH' ? 'danger' : row.urgency === 'MEDIUM' ? 'warning' : 'info'">{{ row.urgency }}</el-tag></template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Anomaly Detection -->
      <el-tab-pane label="Anomaly Detection" name="anomalies">
        <div class="section-toolbar">
          <el-button type="primary" size="small" @click="loadAnomalies">Query</el-button>
        </div>
        <el-table :data="anomalyResult" border stripe>
          <el-table-column prop="type" label="Type" width="100" />
          <el-table-column prop="description" label="Description" min-width="250" />
          <el-table-column prop="severity" label="Severity" width="100">
            <template #default="{ row }"><el-tag :type="row.severity === 'HIGH' ? 'danger' : row.severity === 'MEDIUM' ? 'warning' : 'info'">{{ row.severity }}</el-tag></template>
          </el-table-column>
          <el-table-column prop="detectedAt" label="Detected At" width="170" />
        </el-table>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.aw-container { padding: 20px; }
.toolbar { margin-bottom: 16px; }
.stat-value { font-size: 28px; font-weight: bold; color: #303133; text-align: center; }
.stat-label { font-size: 14px; color: #909399; text-align: center; margin-top: 5px; }
.section-toolbar { margin-bottom: 12px; }
</style>
