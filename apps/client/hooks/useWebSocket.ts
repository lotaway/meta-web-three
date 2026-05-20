import { useEffect, useRef, useState, useCallback } from 'react'
import { API_BASE_URL } from '@/api/generated'

type MessageHandler = (data: any) => void

export function useWebSocket(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const handlersRef = useRef<Map<string, MessageHandler[]>>(new Map())
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const retryCountRef = useRef(0)

  const wsBaseUrl = API_BASE_URL.replace(/^http/, 'ws')

  const connect = useCallback(() => {
    if (!sessionId || wsRef.current?.readyState === WebSocket.OPEN) return
    const url = `${wsBaseUrl}/ws/cs/${sessionId}`
    const ws = new WebSocket(url)
    ws.onopen = () => {
      setConnected(true)
      retryCountRef.current = 0
      ws.send(JSON.stringify({ type: 'REGISTER_CUSTOMER' }))
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'PING' }))
      }, 30000)
    }
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const type = data.type
        const handlers = handlersRef.current.get(type)
        if (handlers) handlers.forEach((h) => h(data))
      } catch { }
    }
    ws.onclose = () => {
      setConnected(false)
      if (pingTimerRef.current) clearInterval(pingTimerRef.current)
      const delay = Math.min(1000 * Math.pow(2, retryCountRef.current), 30000)
      retryCountRef.current++
      retryRef.current = setTimeout(connect, delay)
    }
    ws.onerror = () => { ws.close() }
    wsRef.current = ws
  }, [sessionId, wsBaseUrl])

  useEffect(() => {
    connect()
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current)
      if (pingTimerRef.current) clearInterval(pingTimerRef.current)
      if (wsRef.current) wsRef.current.close()
      wsRef.current = null
      setConnected(false)
    }
  }, [connect])

  const send = useCallback((data: Record<string, any>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const on = useCallback((type: string, handler: MessageHandler) => {
    const handlers = handlersRef.current.get(type) || []
    handlers.push(handler)
    handlersRef.current.set(type, handlers)
    return () => {
      const current = handlersRef.current.get(type) || []
      handlersRef.current.set(type, current.filter((h) => h !== handler))
    }
  }, [])

  return { connected, send, on }
}
