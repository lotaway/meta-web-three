<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import {
  listPendingDevelopersAPI, listApprovedDevelopersAPI,
  approveDeveloperAPI, rejectDeveloperAPI, suspendDeveloperAPI, reactivateDeveloperAPI,
  type Developer,
  listDeveloperApiKeysAPI, createApiKeyAPI, disableApiKeyAPI, enableApiKeyAPI, revokeApiKeyAPI,
  type ApiKey,
  listPendingSubscriptionsAPI, listActiveSubscriptionsAPI,
  createSubscriptionAPI, approveSubscriptionAPI, rejectSubscriptionAPI, cancelSubscriptionAPI,
  type Subscription,
  listDeveloperOAuthAppsAPI, createOAuthAppAPI, disableOAuthAppAPI, enableOAuthAppAPI, deleteOAuthAppAPI,
  type OAuthApp,
  getCurrentUsageAPI, getDocsOpenAPIAPI, getSDKSamplesAPI,
} from '@/apis/developer-portal'

const activeTab = ref('developers')
const loading = ref(false)

// Developers
const pendingDevs = ref<Developer[]>([])
const approvedDevs = ref<Developer[]>([])
const loadDevelopers = async () => {
  try {
    const [p, a] = await Promise.allSettled([listPendingDevelopersAPI(), listApprovedDevelopersAPI()])
    if (p.status === 'fulfilled') pendingDevs.value = (p.value.data as any) || []
    if (a.status === 'fulfilled') approvedDevs.value = (a.value.data as any) || []
  } catch (e: any) { console.error('[loadDevelopers]', e); ElMessage.error('加载失败') }
}

const approveDev = async (id: string) => {
  try {
    const { value } = await ElMessageBox.prompt('Reviewer name:', 'Approve Developer')
    if (!value) return
    await approveDeveloperAPI(id, value)
    ElMessage.success('Approved')
    loadDevelopers()
  } catch (e: any) { if (e !== 'cancel') { console.error(e); ElMessage.error('操作失败') } }
}

const rejectDev = async (id: string) => {
  try {
    const { value } = await ElMessageBox.prompt('Reason:', 'Reject Developer')
    if (!value) return
    await rejectDeveloperAPI(id, 'admin', value)
    ElMessage.success('Rejected')
    loadDevelopers()
  } catch (e: any) { if (e !== 'cancel') { console.error(e); ElMessage.error('操作失败') } }
}

const suspendDev = async (id: string) => {
  try {
    const { value } = await ElMessageBox.prompt('Reason:', 'Suspend Developer')
    if (!value) return
    await suspendDeveloperAPI(id, value)
    ElMessage.success('Suspended')
    loadDevelopers()
  } catch (e: any) { if (e !== 'cancel') { console.error(e); ElMessage.error('操作失败') } }
}

// API Keys
const selectedDevId = ref('')
const apiKeys = ref<ApiKey[]>([])
const loadApiKeys = async (devId: string) => {
  selectedDevId.value = devId
  if (!devId) { apiKeys.value = []; return }
  try { const res = await listDeveloperApiKeysAPI(devId); apiKeys.value = (res.data as any) || [] } catch (e: any) { console.error('[loadApiKeys]', e); ElMessage.error('加载失败') }
}

const keyDialog = ref(false)
const keyForm = ref({ developerId: '', keyName: '', scopes: '' })
const createKey = async () => {
  try {
    await createApiKeyAPI(keyForm.value.developerId, { keyName: keyForm.value.keyName, scopes: keyForm.value.scopes?.split(',').map(s => s.trim()) || [] })
    ElMessage.success('Key created')
    keyDialog.value = false
    loadApiKeys(keyForm.value.developerId)
  } catch (_) { ElMessage.error('Failed') }
}

// Subscriptions
const pendingSubs = ref<Subscription[]>([])
const activeSubs = ref<Subscription[]>([])
const loadSubscriptions = async () => {
  try {
    const [p, a] = await Promise.allSettled([listPendingSubscriptionsAPI(), listActiveSubscriptionsAPI()])
    if (p.status === 'fulfilled') pendingSubs.value = (p.value.data as any) || []
    if (a.status === 'fulfilled') activeSubs.value = (a.value.data as any) || []
  } catch (e: any) { console.error('[loadSubscriptions]', e); ElMessage.error('加载失败') }
}

const approveSub = async (id: string) => {
  try {
    const { value } = await ElMessageBox.prompt('Reviewer:', 'Approve')
    if (!value) return
    await approveSubscriptionAPI(id, value)
    ElMessage.success('Approved')
    loadSubscriptions()
  } catch (e: any) { if (e !== 'cancel') { console.error(e); ElMessage.error('操作失败') } }
}

const rejectSub = async (id: string) => {
  try {
    const { value } = await ElMessageBox.prompt('Reason:', 'Reject')
    if (!value) return
    await rejectSubscriptionAPI(id, 'admin', value)
    ElMessage.success('Rejected')
    loadSubscriptions()
  } catch (e: any) { if (e !== 'cancel') { console.error(e); ElMessage.error('操作失败') } }
}

// OAuth Apps
const selectedOAuthDevId = ref('')
const oauthApps = ref<OAuthApp[]>([])
const loadOAuthApps = async (devId: string) => {
  selectedOAuthDevId.value = devId
  if (!devId) { oauthApps.value = []; return }
  try { const res = await listDeveloperOAuthAppsAPI(devId); oauthApps.value = (res.data as any) || [] } catch (e: any) { console.error('[loadOAuthApps]', e); ElMessage.error('加载失败') }
}

const oauthDialog = ref(false)
const oauthForm = ref({ developerId: '', appName: '', redirectUris: '' })
const createOAuth = async () => {
  try {
    await createOAuthAppAPI(oauthForm.value.developerId, { appName: oauthForm.value.appName, redirectUris: oauthForm.value.redirectUris.split(',').map(s => s.trim()) })
    ElMessage.success('OAuth app created')
    oauthDialog.value = false
    loadOAuthApps(oauthForm.value.developerId)
  } catch (_) { ElMessage.error('Failed') }
}

// Usage
const currentUsage = ref<Record<string, any>>({})
const usageDevId = ref('')
const loadUsage = async (devId: string) => {
  usageDevId.value = devId
  if (!devId) { currentUsage.value = {}; return }
  try { const res = await getCurrentUsageAPI(devId); currentUsage.value = (res.data as any) || {} } catch (e: any) { console.error('[loadUsage]', e); ElMessage.error('加载失败') }
}

// Docs
const docsOpenApi = ref<Record<string, any>>({})
const sdkSamples = ref<Record<string, string>>({})
const loadDocs = async () => {
  try {
    const [d, s] = await Promise.allSettled([getDocsOpenAPIAPI(), getSDKSamplesAPI()])
    if (d.status === 'fulfilled') docsOpenApi.value = (d.value.data as any) || {}
    if (s.status === 'fulfilled') sdkSamples.value = (s.value.data as any) || {}
  } catch (e: any) { console.error('[loadDocs]', e); ElMessage.error('加载失败') }
}

const refreshAll = () => { loadDevelopers(); loadSubscriptions(); loadDocs() }

const statusTag = (s?: string) => {
  const m: Record<string, string> = { PENDING: 'info', APPROVED: 'success', REJECTED: 'danger', ACTIVE: 'success', SUSPENDED: 'warning', INACTIVE: 'info', CANCELLED: 'info' }
  return m[s || ''] || 'info'
}

onMounted(refreshAll)
</script>

<template>
  <div class="dp-container">
    <div class="toolbar">
      <el-button :icon="Refresh" @click="refreshAll">Refresh</el-button>
    </div>

    <el-tabs v-model="activeTab" type="border-card">
      <!-- Developers -->
      <el-tab-pane label="Developers" name="developers">
        <el-tabs>
          <el-tab-pane label="Pending Approval" name="pending">
            <el-table :data="pendingDevs" border stripe v-loading="loading">
              <el-table-column prop="developerId" label="ID" width="160" />
              <el-table-column prop="email" label="Email" min-width="180" />
              <el-table-column prop="companyName" label="Company" min-width="150" />
              <el-table-column prop="contactName" label="Contact" width="120" />
              <el-table-column prop="status" label="Status" width="100"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
              <el-table-column prop="createdAt" label="Registered" width="170" />
              <el-table-column label="Actions" width="180" fixed="right">
                <template #default="{ row }">
                  <el-button link type="success" size="small" @click="approveDev(row.developerId)">Approve</el-button>
                  <el-button link type="danger" size="small" @click="rejectDev(row.developerId)">Reject</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
          <el-tab-pane label="Approved" name="approved">
            <el-table :data="approvedDevs" border stripe>
              <el-table-column prop="developerId" label="ID" width="160" />
              <el-table-column prop="email" label="Email" min-width="180" />
              <el-table-column prop="companyName" label="Company" min-width="150" />
              <el-table-column prop="plan" label="Plan" width="100" />
              <el-table-column prop="status" label="Status" width="100"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
              <el-table-column label="Actions" width="140">
                <template #default="{ row }">
                  <el-button v-if="row.status === 'APPROVED'" link type="warning" size="small" @click="suspendDev(row.developerId)">Suspend</el-button>
                  <el-button v-if="row.status === 'SUSPENDED'" link type="success" size="small" @click="reactivateDeveloperAPI(row.developerId).then(loadDevelopers); ElMessage.success('Reactivated')">Reactivate</el-button>
                  <el-button link type="primary" size="small" @click="loadApiKeys(row.developerId); activeTab = 'api-keys'">Keys</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </el-tab-pane>

      <!-- API Keys -->
      <el-tab-pane label="API Keys" name="api-keys">
        <div class="section-toolbar">
          <el-select v-model="selectedDevId" placeholder="Select developer" @change="loadApiKeys" style="width:220px; margin-right:8px">
            <el-option v-for="d in approvedDevs" :key="d.developerId" :label="d.email" :value="d.developerId" />
          </el-select>
          <el-button v-if="selectedDevId" type="primary" @click="keyForm.developerId = selectedDevId; keyDialog = true">Create Key</el-button>
        </div>
        <el-table :data="apiKeys" border stripe>
          <el-table-column prop="keyId" label="Key ID" width="180" />
          <el-table-column prop="keyName" label="Name" min-width="150" />
          <el-table-column prop="status" label="Status" width="90"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
          <el-table-column prop="createdAt" label="Created" width="170" />
          <el-table-column prop="expiresAt" label="Expires" width="170" />
          <el-table-column label="Actions" width="240" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="disableApiKeyAPI(row.keyId).then(() => loadApiKeys(selectedDevId)); ElMessage.success('Disabled')">Disable</el-button>
              <el-button v-if="row.status === 'INACTIVE'" link type="success" size="small" @click="enableApiKeyAPI(row.keyId).then(() => loadApiKeys(selectedDevId)); ElMessage.success('Enabled')">Enable</el-button>
              <el-button link type="danger" size="small" @click="revokeApiKeyAPI(row.keyId).then(() => loadApiKeys(selectedDevId)); ElMessage.success('Revoked')">Revoke</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Subscriptions -->
      <el-tab-pane label="Subscriptions" name="subscriptions">
        <el-tabs>
          <el-tab-pane label="Pending" name="sub-pending">
            <el-table :data="pendingSubs" border stripe>
              <el-table-column prop="subscriptionId" label="ID" width="160" />
              <el-table-column prop="developerId" label="Developer" width="160" />
              <el-table-column prop="apiName" label="API" min-width="150" />
              <el-table-column prop="tier" label="Tier" width="80" />
              <el-table-column prop="status" label="Status" width="90"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
              <el-table-column label="Actions" width="160">
                <template #default="{ row }">
                  <el-button link type="success" size="small" @click="approveSub(row.subscriptionId)">Approve</el-button>
                  <el-button link type="danger" size="small" @click="rejectSub(row.subscriptionId)">Reject</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
          <el-tab-pane label="Active" name="sub-active">
            <el-table :data="activeSubs" border stripe>
              <el-table-column prop="subscriptionId" label="ID" width="160" />
              <el-table-column prop="developerId" label="Developer" width="160" />
              <el-table-column prop="apiName" label="API" min-width="150" />
              <el-table-column prop="tier" label="Tier" width="80" />
              <el-table-column prop="status" label="Status" width="90"><template #default="{ row }"><el-tag type="success" size="small">{{ row.status }}</el-tag></template></el-table-column>
              <el-table-column label="Actions" width="100">
                <template #default="{ row }">
                  <el-button link type="warning" size="small" @click="cancelSubscriptionAPI(row.subscriptionId).then(loadSubscriptions); ElMessage.success('Cancelled')">Cancel</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </el-tab-pane>

      <!-- OAuth Apps -->
      <el-tab-pane label="OAuth Apps" name="oauth">
        <div class="section-toolbar">
          <el-select v-model="selectedOAuthDevId" placeholder="Select developer" @change="loadOAuthApps" style="width:220px; margin-right:8px">
            <el-option v-for="d in approvedDevs" :key="d.developerId" :label="d.email" :value="d.developerId" />
          </el-select>
          <el-button v-if="selectedOAuthDevId" type="primary" @click="oauthForm.developerId = selectedOAuthDevId; oauthDialog = true">Create App</el-button>
        </div>
        <el-table :data="oauthApps" border stripe>
          <el-table-column prop="clientId" label="Client ID" width="180" />
          <el-table-column prop="appName" label="App Name" min-width="160" />
          <el-table-column prop="redirectUris" label="Redirect URIs" min-width="200" show-overflow-tooltip />
          <el-table-column prop="status" label="Status" width="90"><template #default="{ row }"><el-tag :type="statusTag(row.status)" size="small">{{ row.status }}</el-tag></template></el-table-column>
          <el-table-column label="Actions" width="200">
            <template #default="{ row }">
              <el-button v-if="row.status === 'ACTIVE'" link type="warning" size="small" @click="disableOAuthAppAPI(row.clientId).then(() => loadOAuthApps(selectedOAuthDevId)); ElMessage.success('Disabled')">Disable</el-button>
              <el-button v-if="row.status === 'INACTIVE'" link type="success" size="small" @click="enableOAuthAppAPI(row.clientId).then(() => loadOAuthApps(selectedOAuthDevId)); ElMessage.success('Enabled')">Enable</el-button>
              <el-button link type="danger" size="small" @click="deleteOAuthAppAPI(row.clientId).then(() => loadOAuthApps(selectedOAuthDevId)); ElMessage.success('Deleted')">Delete</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <!-- Usage -->
      <el-tab-pane label="Usage Stats" name="usage">
        <div class="section-toolbar">
          <el-select v-model="usageDevId" placeholder="Select developer" @change="loadUsage" style="width:220px">
            <el-option v-for="d in approvedDevs" :key="d.developerId" :label="d.email" :value="d.developerId" />
          </el-select>
        </div>
        <el-descriptions v-if="usageDevId && Object.keys(currentUsage).length" border column="2">
          <el-descriptions-item v-for="(v, k) in currentUsage" :key="k" :label="k">{{ v }}</el-descriptions-item>
        </el-descriptions>
        <el-empty v-else-if="usageDevId && !Object.keys(currentUsage).length" description="No usage data" />
        <el-empty v-else description="Select a developer to view usage" />
      </el-tab-pane>

      <!-- Docs -->
      <el-tab-pane label="Documentation" name="docs">
        <el-descriptions border column="1">
          <el-descriptions-item label="OpenAPI Spec">{{ docsOpenApi.openapi || 'Not loaded' }}</el-descriptions-item>
          <el-descriptions-item label="Info Title">{{ docsOpenApi.info?.title || '-' }}</el-descriptions-item>
          <el-descriptions-item label="Info Version">{{ docsOpenApi.info?.version || '-' }}</el-descriptions-item>
        </el-descriptions>
        <h4 style="margin:16px 0 8px">SDK Samples</h4>
        <el-descriptions border column="1">
          <el-descriptions-item v-for="(v, k) in sdkSamples" :key="k" :label="k">{{ v }}</el-descriptions-item>
        </el-descriptions>
        <el-empty v-if="!Object.keys(sdkSamples).length" description="No SDK samples available" />
      </el-tab-pane>
    </el-tabs>

    <!-- Dialogs -->
    <el-dialog v-model="keyDialog" title="Create API Key" width="400px">
      <el-form :model="keyForm" label-width="110px">
        <el-form-item label="Key Name"><el-input v-model="keyForm.keyName" /></el-form-item>
        <el-form-item label="Scopes"><el-input v-model="keyForm.scopes" placeholder="read,write" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="keyDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createKey">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="oauthDialog" title="Create OAuth App" width="450px">
      <el-form :model="oauthForm" label-width="120px">
        <el-form-item label="App Name"><el-input v-model="oauthForm.appName" /></el-form-item>
        <el-form-item label="Redirect URIs"><el-input v-model="oauthForm.redirectUris" placeholder="https://example.com/callback" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="oauthDialog = false">Cancel</el-button>
        <el-button type="primary" @click="createOAuth">Create</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.dp-container { padding: 20px; }
.toolbar { margin-bottom: 16px; }
.section-toolbar { margin-bottom: 12px; }
</style>
