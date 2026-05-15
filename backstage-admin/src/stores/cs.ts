import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useCsStore = defineStore('cs', () => {
  const wsConnected = ref(false)
  const ws = ref<WebSocket | null>(null)
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null

  function connect(agentId: number) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) return
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = import.meta.env.VITE_WS_HOST || window.location.hostname
    const port = import.meta.env.VITE_WS_PORT || '10091'
    const url = `${protocol}//${host}:${port}/ws/cs/agent-${agentId}`
    const socket = new WebSocket(url)
    socket.onopen = () => {
      wsConnected.value = true
      socket.send(JSON.stringify({ type: 'REGISTER_AGENT', agentId }))
      heartbeatTimer = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'PING' }))
        }
      }, 30000)
    }
    socket.onclose = () => {
      wsConnected.value = false
      if (heartbeatTimer) clearInterval(heartbeatTimer)
      setTimeout(() => connect(agentId), 3000)
    }
    socket.onerror = () => {
      wsConnected.value = false
    }
    ws.value = socket
  }

  function disconnect() {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
    if (ws.value) ws.value.close()
    ws.value = null
    wsConnected.value = false
  }

  function sendMessage(sessionId: string, msgType: string, content: string) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(JSON.stringify({ type: 'SEND_MESSAGE', sessionId, msgType, content }))
    }
  }

  return { wsConnected, connect, disconnect, sendMessage }
})
