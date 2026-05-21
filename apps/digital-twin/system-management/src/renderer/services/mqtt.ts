import { useEffect, useRef, useState, useCallback } from 'react'

type MessageHandler = (topic: string, message: any) => void

interface MQTTServiceOptions {
  brokerUrl: string
  clientId?: string
  username?: string
  password?: string
  topics?: string[]
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
  cleanSession?: boolean
  reconnectPeriod?: number
}

// Simple MQTT-like implementation using WebSocket
// Note: In production, you'd use mqtt.js library
class DigitalTwinMQTT {
  private ws: WebSocket | null = null
  private brokerUrl: string
  private clientId: string
  private username?: string
  private password?: string
  private topics: string[] = []
  private handlers: Map<string, MessageHandler[]> = new Map()
  private isConnected = false
  private isManualClose = false
  private reconnectTimer?: NodeJS.Timeout
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private onConnectCallback?: () => void
  private onDisconnectCallback?: () => void
  private onErrorCallback?: (error: Error) => void

  constructor(options: MQTTServiceOptions) {
    this.brokerUrl = options.brokerUrl
    this.clientId = options.clientId || `dt-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    this.username = options.username
    this.password = options.password
    this.topics = options.topics || []
    this.onConnectCallback = options.onConnect
    this.onDisconnectCallback = options.onDisconnect
    this.onErrorCallback = options.onError
  }

  async connect() {
    return new Promise<void>((resolve, reject) => {
      try {
        // Using a mock WebSocket connection for demo
        // In production, you'd connect to an actual MQTT broker via WebSocket
        this.ws = new WebSocket(this.brokerUrl)

        this.ws.onopen = () => {
          console.log('[MQTT] Connected to broker')
          this.isConnected = true
          this.reconnectAttempts = 0
          this.onConnectCallback?.()
          
          // Subscribe to default topics
          this.topics.forEach(topic => this.subscribe(topic))
          resolve()
        }

        this.ws.onclose = () => {
          console.log('[MQTT] Disconnected from broker')
          this.isConnected = false
          this.onDisconnectCallback?.()
          
          if (!this.isManualClose) {
            this.attemptReconnect()
          }
        }

        this.ws.onerror = (error) => {
          console.error('[MQTT] Error:', error)
          this.onErrorCallback?.(new Error('MQTT connection error'))
          reject(error)
        }

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (e) {
            console.error('[MQTT] Failed to parse message:', e)
          }
        }
      } catch (error) {
        console.error('[MQTT] Connection failed:', error)
        reject(error)
      }
    })
  }

  private handleMessage(message: any) {
    const { topic, payload } = message
    
    // Notify specific topic handlers
    const handlers = this.handlers.get(topic) || []
    handlers.forEach(handler => handler(topic, payload))
    
    // Notify wildcard handlers
    const wildcardHandlers = this.handlers.get('#') || []
    wildcardHandlers.forEach(handler => handler(topic, payload))
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000)
      console.log(`[MQTT] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)
      this.reconnectTimer = setTimeout(() => this.connect(), delay)
    } else {
      console.error('[MQTT] Max reconnection attempts reached')
    }
  }

  disconnect() {
    this.isManualClose = true
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.isConnected = false
  }

  subscribe(topic: string, handler?: MessageHandler) {
    if (!this.topics.includes(topic)) {
      this.topics.push(topic)
    }

    if (handler) {
      const handlers = this.handlers.get(topic) || []
      handlers.push(handler)
      this.handlers.set(topic, handlers)
    }

    // Send subscription message to broker
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        topic
      }))
    }
  }

  unsubscribe(topic: string, handler?: MessageHandler) {
    if (handler) {
      const handlers = this.handlers.get(topic) || []
      const index = handlers.indexOf(handler)
      if (index > -1) {
        handlers.splice(index, 1)
      }
    }

    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        topic
      }))
    }
  }

  publish(topic: string, payload: any) {
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify({
        type: 'publish',
        topic,
        payload
      }))
    } else {
      console.warn('[MQTT] Cannot publish, not connected')
    }
  }

  getTopics() {
    return [...this.topics]
  }

  getConnectionStatus() {
    return this.isConnected
  }
}

// React Hook for MQTT
export function useDigitalTwinMQTT(options: MQTTServiceOptions) {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const mqttRef = useRef<DigitalTwinMQTT | null>(null)

  useEffect(() => {
    const mqtt = new DigitalTwinMQTT({
      ...options,
      onConnect: () => setIsConnected(true),
      onDisconnect: () => setIsConnected(false),
      onError: (err) => setError(err)
    })

    mqtt.connect().catch(err => console.error('[MQTT] Connect error:', err))
    mqttRef.current = mqtt

    return () => {
      mqtt.disconnect()
    }
  }, [options.brokerUrl])

  const subscribe = useCallback((topic: string, handler: MessageHandler) => {
    mqttRef.current?.subscribe(topic, handler)
    return () => mqttRef.current?.unsubscribe(topic, handler)
  }, [])

  const publish = useCallback((topic: string, payload: any) => {
    mqttRef.current?.publish(topic, payload)
  }, [])

  return { isConnected, error, subscribe, publish }
}

// Device data types
export interface DeviceTelemetry {
  deviceCode: string
  timestamp: number
  metrics: {
    temperature?: number
    humidity?: number
    pressure?: number
    vibration?: number
    power?: number
    rpm?: number
  }
  location?: {
    x: number
    y: number
    z: number
  }
}

export interface DeviceStatusMessage {
  deviceCode: string
  status: 'online' | 'offline' | 'running' | 'idle' | 'warning' | 'error'
  timestamp: number
}

// Predefined topic patterns for digital twin
export const DT_TOPICS = {
  DEVICE_STATUS: 'device/+/status',
  DEVICE_TELEMETRY: 'device/+/telemetry',
  DEVICE_POSITION: 'device/+/position',
  ALERT_CREATED: 'alert/created',
  ALERT_UPDATED: 'alert/updated',
  PRODUCTION_OUTPUT: 'production/+/output',
  AGV_POSITION: 'agv/+/position'
}

export { DigitalTwinMQTT }
export type { MQTTServiceOptions, MessageHandler }