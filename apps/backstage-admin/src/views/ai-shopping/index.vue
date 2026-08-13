<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh, MagicStick } from '@element-plus/icons-vue'
import {
  listAiShoppingConfigsAPI,
  saveAiShoppingConfigAPI,
  deleteAiShoppingConfigAPI,
  rebuildAiShoppingIndexAPI,
  getAiShoppingIndexStatusAPI,
  testAiProviderAPI,
  getAiShoppingLogsAPI,
  type AiShoppingConfig,
  type AiSearchLog,
  type AiIndexStatus,
} from '@/apis/aiShopping'

const activeTab = ref('config')
const loading = ref(false)

// ==================== Config ====================
const configs = ref<AiShoppingConfig[]>([])
const configForm = ref({ configKey: '', configValue: '', description: '' })

const loadConfigs = async () => {
  try {
    const res = await listAiShoppingConfigsAPI()
    configs.value = (res.data as AiShoppingConfig[]) || []
  } catch (e) {
    console.error('[loadConfigs]', e)
    ElMessage.error('加载配置失败')
  }
}

const saveConfig = async () => {
  if (!configForm.value.configKey.trim()) {
    ElMessage.warning('请填写配置项 key')
    return
  }
  try {
    await saveAiShoppingConfigAPI(configForm.value)
    ElMessage.success('配置已保存')
    configForm.value = { configKey: '', configValue: '', description: '' }
    loadConfigs()
  } catch (e) {
    console.error('[saveConfig]', e)
  }
}

const removeConfig = async (row: AiShoppingConfig) => {
  try {
    await ElMessageBox.confirm(`确定删除配置 ${row.configKey} ？`, '提示', { type: 'warning' })
    await deleteAiShoppingConfigAPI(row.configKey)
    ElMessage.success('配置已删除')
    loadConfigs()
  } catch (e) {
    if ((e as { message?: string }).message) return
    console.error('[removeConfig]', e)
  }
}

const suggestKey = (example: string) => {
  configForm.value.configKey = example
}

// ==================== Index ====================
const indexStatus = ref<AiIndexStatus | null>(null)
const indexLoading = ref(false)

const loadIndexStatus = async () => {
  try {
    const res = await getAiShoppingIndexStatusAPI()
    indexStatus.value = res.data as AiIndexStatus
  } catch (e) {
    console.error('[loadIndexStatus]', e)
    ElMessage.error('获取索引状态失败')
  }
}

const rebuildIndex = async (type: string) => {
  indexLoading.value = true
  try {
    const res = await rebuildAiShoppingIndexAPI(type)
    ElMessage.success(`索引重建任务已启动 (${res.data?.type})`)
    setTimeout(loadIndexStatus, 3000)
  } catch (e) {
    console.error('[rebuildIndex]', e)
  } finally {
    indexLoading.value = false
  }
}

// ==================== Provider test ====================
const providerTesting = ref('')
const providerResults = ref<Record<string, string>>({})

const testProvider = async (type: string) => {
  providerTesting.value = type
  try {
    const res = await testAiProviderAPI(type)
    const data = res.data
    if (data?.success) {
      providerResults.value[type] = `成功 (${data.responseTimeMs}ms)`
      ElMessage.success(`Provider ${type} 连接成功`)
    } else {
      providerResults.value[type] = `失败: ${data?.error || 'unknown'}`
      ElMessage.error(`Provider ${type} 连接失败`)
    }
  } catch (e) {
    console.error('[testProvider]', e)
    providerResults.value[type] = '请求失败'
  } finally {
    providerTesting.value = ''
  }
}

// ==================== Logs ====================
const logs = ref<AiSearchLog[]>([])
const loadLogs = async () => {
  try {
    const res = await getAiShoppingLogsAPI(50)
    logs.value = (res.data as AiSearchLog[]) || []
  } catch (e) {
    console.error('[loadLogs]', e)
  }
}

const refreshAll = async () => {
  loading.value = true
  await Promise.all([loadConfigs(), loadIndexStatus(), loadLogs()])
  loading.value = false
}

const searchTypeLabels: Record<string, string> = {
  TEXT_CORRECT: '文本纠错',
  SMART_MATCH: '智能匹配',
  IMAGE_SEARCH: '以图搜图',
  COMBINED_SEARCH: '一站式搜索',
}

onMounted(() => {
  loadConfigs()
  loadIndexStatus()
  loadLogs()
})
</script>

<template>
  <div class="app-container">
    <div class="page-header">
      <div class="page-header-left">
        <h2 class="page-title">
          <el-icon><MagicStick /></el-icon>
          AI 辅助购物
        </h2>
        <span class="page-subtitle">文本纠错 / 智能匹配 / 以图搜图</span>
      </div>
      <el-button type="primary" :icon="Refresh" :loading="loading" @click="refreshAll">刷新</el-button>
    </div>

    <el-tabs v-model="activeTab" class="ai-tabs">
      <!-- ==================== 配置 ==================== -->
      <el-tab-pane label="Provider 配置" name="config">
        <el-row :gutter="16">
          <el-col :span="10">
            <el-card shadow="never" class="mb-16">
              <template #header>
                <span>新增 / 覆盖配置项（DB 覆盖 application.yml，格式 key=value）</span>
              </template>
              <el-form label-width="90px">
                <el-form-item label="Key">
                  <el-input v-model="configForm.configKey" placeholder="如 embedding.base-url" />
                </el-form-item>
                <el-form-item label="Value">
                  <el-input v-model="configForm.configValue" placeholder="配置值" />
                </el-form-item>
                <el-form-item label="说明">
                  <el-input v-model="configForm.description" placeholder="用途说明（可选）" />
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" @click="saveConfig">保存</el-button>
                  <el-button
                    v-for="key in ['embedding.base-url', 'embedding.api-key', 'embedding.model', 'llm.base-url', 'llm.api-key', 'llm.model', 'image-embedding.base-url', 'image-embedding.api-key', 'image-embedding.model', 'vector.store', 'milvus.host', 'milvus.port']"
                    :key="key"
                    size="small"
                    @click="suggestKey(key)"
                  >{{ key }}</el-button>
                </el-form-item>
              </el-form>
            </el-card>
          </el-col>
          <el-col :span="14">
            <el-card shadow="never">
              <template #header>
                <span>已配置项</span>
              </template>
              <el-table :data="configs" border size="small">
                <el-table-column prop="configKey" label="Key" min-width="200" />
                <el-table-column prop="configValue" label="Value" min-width="140" show-overflow-tooltip />
                <el-table-column prop="description" label="说明" min-width="120" show-overflow-tooltip />
                <el-table-column label="操作" width="80" align="center">
                  <template #default="{ row }">
                    <el-button link type="danger" @click="removeConfig(row)">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </el-col>
        </el-row>
      </el-tab-pane>

      <!-- ==================== 索引 ==================== -->
      <el-tab-pane label="向量索引" name="index">
        <el-card shadow="never" class="mb-16">
          <template #header>
            <span>Milvus 向量索引状态</span>
          </template>
          <el-descriptions :column="2" border v-if="indexStatus">
            <el-descriptions-item label="向量存储">
              <el-tag>{{ indexStatus.vectorStore }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="文本集合">{{ indexStatus.collectionText }}</el-descriptions-item>
            <el-descriptions-item label="图像集合">{{ indexStatus.collectionImage }}</el-descriptions-item>
            <el-descriptions-item label="状态">
              <el-tag :type="String(indexStatus.status?.status || '').toLowerCase().includes('idle') ? 'success' : 'warning'">
                {{ (indexStatus.status?.status as string) || '-' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="商品数" :span="2">
              {{ String(indexStatus.status?.productCount ?? '-') }}
            </el-descriptions-item>
            <el-descriptions-item label="最近重建" :span="2">
              {{ String(indexStatus.status?.lastRebuiltAt ?? '-') }}
            </el-descriptions-item>
            <el-descriptions-item label="错误信息" :span="2">
              {{ String(indexStatus.status?.error ?? '-') }}
            </el-descriptions-item>
          </el-descriptions>
          <div class="mt-16">
            <el-button type="primary" :loading="indexLoading" @click="rebuildIndex('all')">重建全部索引</el-button>
            <el-button :loading="indexLoading" @click="rebuildIndex('text')">仅重建文本索引</el-button>
            <el-button :loading="indexLoading" @click="rebuildIndex('image')">仅重建图像索引</el-button>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- ==================== Provider 测试 ==================== -->
      <el-tab-pane label="Provider 测试" name="provider">
        <el-card shadow="never">
          <template #header>
            <span>AI Provider 连通性测试</span>
          </template>
          <div class="provider-row">
            <el-button
              v-for="t in ['embedding', 'image', 'llm']"
              :key="t"
              :loading="providerTesting === t"
              @click="testProvider(t)"
            >
              {{ t === 'embedding' ? '测试文本嵌入' : t === 'image' ? '测试图像嵌入' : '测试 LLM' }}
            </el-button>
            <span
              v-for="t in ['embedding', 'image', 'llm']"
              :key="`r-${t}`"
              v-show="providerResults[t]"
              class="provider-result"
              :class="{ error: providerResults[t].startsWith('失败') }"
            >
              [{{ t }}] {{ providerResults[t] }}
            </span>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- ==================== 日志 ==================== -->
      <el-tab-pane label="搜索日志" name="logs">
        <el-card shadow="never">
          <template #header>
            <span>最近 AI 搜索日志（50 条）</span>
          </template>
          <el-table :data="logs" border size="small">
            <el-table-column prop="id" label="ID" width="70" />
            <el-table-column label="类型" width="110">
              <template #default="{ row }">
                <el-tag size="small">{{ searchTypeLabels[row.searchType] || row.searchType }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="userId" label="用户" width="80">
              <template #default="{ row }">{{ row.userId ?? '-' }}</template>
            </el-table-column>
            <el-table-column prop="queryText" label="原始词" min-width="140" show-overflow-tooltip />
            <el-table-column prop="correctedText" label="纠错词" min-width="140" show-overflow-tooltip />
            <el-table-column prop="resultCount" label="结果数" width="80" align="center" />
            <el-table-column prop="responseTimeMs" label="耗时(ms)" width="90" align="center" />
            <el-table-column prop="createdAt" label="时间" width="180" />
          </el-table>
        </el-card>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.app-container {
  padding: 16px;
}
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.page-header-left {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.page-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  margin: 0;
}
.page-subtitle {
  color: #909399;
  font-size: 13px;
}
.mb-16 {
  margin-bottom: 16px;
}
.mt-16 {
  margin-top: 16px;
}
.provider-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px dashed #ebeef5;
}
.provider-row:last-child {
  border-bottom: none;
}
.provider-result {
  color: #67c23a;
  font-size: 13px;
}
.provider-result.error {
  color: #f56c6c;
}
</style>
