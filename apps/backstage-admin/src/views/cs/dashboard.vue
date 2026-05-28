<template>
  <div class="cs-dashboard">
    <div class="cs-toolbar">
      <el-select v-model="currentAgentId" placeholder="选择客服身份" @change="switchAgent">
        <el-option v-for="a in agents" :key="a.id" :label="a.nickname" :value="a.id" />
      </el-select>
      <el-tag v-if="csStore.wsConnected" type="success">已连接</el-tag>
      <el-tag v-else type="danger">未连接</el-tag>
      <el-button size="small" @click="toggleStatus">{{ agent?.status === 'ONLINE' ? '下线' : '上线' }}</el-button>
      <span class="cs-load">当前负载: {{ agent?.currentLoad }}/{{ agent?.maxConcurrent }}</span>
    </div>
    <div class="cs-body">
      <div class="cs-sidebar-left">
        <el-tabs>
          <el-tab-pane label="排队中">
            <el-scrollbar>
              <div v-for="c in queuingList" :key="c.sessionId" class="cs-conv-item" @click="selectConversation(c)">
                <div class="cs-conv-header">
                  <span class="cs-conv-user">用户 {{ c.customerId }}</span>
                  <el-tag size="small" type="warning">{{ c.channel }}</el-tag>
                </div>
                <div class="cs-conv-time">{{ formatTime(c.createTime) }}</div>
              </div>
              <el-empty v-if="queuingList.length === 0" description="暂无排队" />
            </el-scrollbar>
          </el-tab-pane>
          <el-tab-pane label="进行中">
            <el-scrollbar>
              <div v-for="c in activeList" :key="c.sessionId" class="cs-conv-item active" @click="selectConversation(c)">
                <div class="cs-conv-header">
                  <span class="cs-conv-user">用户 {{ c.customerId }}</span>
                </div>
                <div class="cs-conv-time">{{ formatTime(c.activeTime) }}</div>
              </div>
              <el-empty v-if="activeList.length === 0" description="暂无进行中" />
            </el-scrollbar>
          </el-tab-pane>
        </el-tabs>
      </div>
      <div class="cs-chat-area">
        <template v-if="currentSession">
          <div class="cs-chat-header">
            会话 {{ currentSession.sessionId.slice(0, 8) }}...
            <el-button size="small" type="danger" plain @click="handleClose">结束会话</el-button>
          </div>
          <el-scrollbar ref="chatScroll" class="cs-messages" @scroll="onScroll">
            <div v-for="msg in messages" :key="msg.messageId" class="cs-msg-row"
              :class="{ 'cs-msg-agent': msg.senderType === 'AGENT', 'cs-msg-customer': msg.senderType === 'CUSTOMER' }">
              <div class="cs-msg-bubble">
                <div class="cs-msg-sender">{{ msg.senderType }}</div>
                <div v-if="msg.msgType === 'PRODUCT_CARD'" class="cs-msg-card">商品卡片: {{ msg.content }}</div>
                <div v-else-if="msg.msgType === 'ORDER_CARD'" class="cs-msg-card">订单卡片: {{ msg.content }}</div>
                <div v-else>{{ msg.content }}</div>
                <div class="cs-msg-time">{{ formatTime(msg.timestamp) }}</div>
              </div>
            </div>
          </el-scrollbar>
          <div class="cs-input-area">
            <el-popover placement="top" trigger="click" width="300">
              <template #reference>
                <el-button size="small">快捷回复</el-button>
              </template>
              <el-scrollbar max-height="250">
                <div v-for="(r, index) in quickReplies" :key="r.id ?? index" class="cs-qr-item" @click="insertQuickReply(r.content)">
                  {{ r.title }}
                </div>
              </el-scrollbar>
            </el-popover>
            <el-input v-model="inputText" type="textarea" :rows="3" placeholder="输入消息..." @keydown.enter.ctrl="handleSend" />
            <el-button type="primary" @click="handleSend">发送</el-button>
          </div>
        </template>
        <el-empty v-else description="请选择会话" />
      </div>
      <div class="cs-sidebar-right">
        <template v-if="currentSession">
          <el-descriptions title="用户信息" :column="1" size="small" border>
            <el-descriptions-item label="用户ID">{{ currentSession.customerId }}</el-descriptions-item>
            <el-descriptions-item label="来源渠道">{{ currentSession.channel }}</el-descriptions-item>
            <el-descriptions-item v-if="currentSession.productId" label="商品ID">{{ currentSession.productId }}</el-descriptions-item>
            <el-descriptions-item v-if="currentSession.orderId" label="订单ID">{{ currentSession.orderId }}</el-descriptions-item>
            <el-descriptions-item label="状态">{{ currentSession.status }}</el-descriptions-item>
            <el-descriptions-item label="创建时间">{{ formatTime(currentSession.createTime) }}</el-descriptions-item>
          </el-descriptions>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useCsStore } from '@/stores/cs'
import {
  getQueuingConversationsAPI, getAgentConversationsAPI, closeConversationAPI,
  getMessagesAPI, sendMessageAPI, agentOnlineAPI, agentOfflineAPI,
  getOnlineAgentsAPI, getAgentInfoAPI, getQuickReplyListAPI,
} from '@/apis/cs'
import type { Conversation, CsMessage, Agent, QuickReply } from '@/apis/cs'
import { ElMessage } from 'element-plus'

const csStore = useCsStore()
const agents = ref<Agent[]>([])
const agent = ref<Agent | null>(null)
const currentAgentId = ref<number>(0)
const queuingList = ref<Conversation[]>([])
const activeList = ref<Conversation[]>([])
const currentSession = ref<Conversation | null>(null)
const messages = ref<CsMessage[]>([])
const inputText = ref('')
const quickReplies = ref<QuickReply[]>([])
const chatScroll = ref<any>(null)

const loadAgents = async () => {
  const res = await getOnlineAgentsAPI()
  agents.value = res.data || []
}

const loadQueuing = async () => {
  const res = await getQueuingConversationsAPI()
  queuingList.value = res.data || []
}

const loadActive = async (agentId: number) => {
  const res = await getAgentConversationsAPI(agentId)
  activeList.value = res.data || []
}

const switchAgent = async (id: number) => {
  csStore.disconnect()
  const res = await getAgentInfoAPI(id)
  agent.value = res.data
  currentAgentId.value = id
  csStore.connect(id)
  loadActive(id)
}

const toggleStatus = async () => {
  if (!agent.value) return
  if (agent.value.status === 'ONLINE') {
    await agentOfflineAPI(agent.value.id)
    agent.value.status = 'OFFLINE'
  } else {
    await agentOnlineAPI(agent.value.id)
    agent.value.status = 'ONLINE'
  }
}

const selectConversation = async (c: Conversation) => {
  currentSession.value = c
  const res = await getMessagesAPI(c.sessionId)
  messages.value = res.data || []
  setTimeout(() => chatScroll.value?.scrollTo(0, 99999), 100)
}

const handleSend = async () => {
  if (!inputText.value.trim() || !currentSession.value || !agent.value) return
  const res = await sendMessageAPI(
    currentSession.value.sessionId, 'AGENT', agent.value.id, 'TEXT', inputText.value)
  if (res.data) messages.value.push(res.data)
  inputText.value = ''
  setTimeout(() => chatScroll.value?.scrollTo(0, 99999), 100)
}

const handleClose = async () => {
  if (!currentSession.value) return
  await closeConversationAPI(currentSession.value.sessionId)
  ElMessage.success('会话已结束')
  currentSession.value = null
  messages.value = []
  if (agent.value) loadActive(agent.value.id)
}

const insertQuickReply = (content: string) => {
  inputText.value = content
}

const formatTime = (t: string) => t ? new Date(t).toLocaleString() : ''

const onScroll = () => {}

onMounted(async () => {
  await loadAgents()
  const res = await getQuickReplyListAPI()
  quickReplies.value = res.data || []
})
</script>

<style scoped>
.cs-dashboard { height: calc(100vh - 100px); display: flex; flex-direction: column; }
.cs-toolbar { display: flex; align-items: center; gap: 12px; padding: 8px 16px; background: #fff; border-bottom: 1px solid #dcdfe6; }
.cs-load { margin-left: auto; font-size: 13px; color: #909399; }
.cs-body { display: flex; flex: 1; overflow: hidden; }
.cs-sidebar-left { width: 260px; border-right: 1px solid #dcdfe6; background: #fff; }
.cs-conv-item { padding: 10px 14px; cursor: pointer; border-bottom: 1px solid #f2f2f2; }
.cs-conv-item:hover { background: #ecf5ff; }
.cs-conv-item.active { background: #f0f9eb; }
.cs-conv-header { display: flex; justify-content: space-between; align-items: center; }
.cs-conv-user { font-weight: 500; font-size: 14px; }
.cs-conv-time { font-size: 12px; color: #c0c4cc; margin-top: 4px; }
.cs-chat-area { flex: 1; display: flex; flex-direction: column; background: #f5f7fa; }
.cs-chat-header { padding: 10px 16px; background: #fff; border-bottom: 1px solid #dcdfe6; display: flex; justify-content: space-between; align-items: center; }
.cs-messages { flex: 1; padding: 16px; }
.cs-msg-row { margin-bottom: 12px; display: flex; }
.cs-msg-agent { justify-content: flex-end; }
.cs-msg-customer { justify-content: flex-start; }
.cs-msg-bubble { max-width: 60%; padding: 8px 12px; border-radius: 8px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.1); }
.cs-msg-agent .cs-msg-bubble { background: #d9ecff; }
.cs-msg-sender { font-size: 11px; color: #909399; margin-bottom: 2px; }
.cs-msg-card { background: #f0f9eb; padding: 6px 10px; border-radius: 4px; font-size: 13px; }
.cs-msg-time { font-size: 11px; color: #c0c4cc; text-align: right; margin-top: 4px; }
.cs-input-area { display: flex; gap: 8px; padding: 12px 16px; background: #fff; border-top: 1px solid #dcdfe6; align-items: flex-end; }
.cs-input-area .el-textarea { flex: 1; }
.cs-sidebar-right { width: 260px; border-left: 1px solid #dcdfe6; background: #fff; padding: 16px; }
.cs-qr-item { padding: 6px 8px; cursor: pointer; border-bottom: 1px solid #f2f2f2; font-size: 13px; }
.cs-qr-item:hover { color: #409eff; }
</style>
