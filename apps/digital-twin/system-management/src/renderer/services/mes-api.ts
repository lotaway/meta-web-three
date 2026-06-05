import axios from 'axios'
import { MES_API_BASE_URL } from '../config/mes'

const client = axios.create({
  baseURL: MES_API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
})

export interface TelemetryMetric {
  metricCode: string
  metricName: string
  value: number
  unit: string
  upperLimit?: number
  lowerLimit?: number
}

export interface TelemetryRecord {
  id: number
  equipmentCode: string
  topic: string
  collectTime: string
  metrics: TelemetryMetric[]
  createdAt: string
}

export interface DeviceCommand {
  id: number
  commandCode: string
  equipmentCode: string
  commandType: string
  payload: string
  status: string
  createdBy: string
  createdAt: string
  executedAt?: string
  resultMessage?: string
}

export interface TraceRecord {
  id?: number
  traceCode: string
  traceType: string
  productCode?: string
  batchNo?: string
  sn?: string
  source?: string
  relations: TraceRelation[]
  createdAt?: string
}

export interface TraceRelation {
  relatedCode: string
  relatedType: string
  relationType: string
  quantity?: number
}

export interface TraceChain {
  root: TraceRecord
  forwardPath: TraceRecord[]
  backwardPath: TraceRecord[]
}

export const mesApi = {
  async fetchTelemetry(equipmentCode: string, limit = 10): Promise<TelemetryRecord[]> {
    const { data } = await client.get<TelemetryRecord[]>(`/api/mes/scada/telemetry/${equipmentCode}`, {
      params: { limit },
    })
    return data ?? []
  },

  async dispatchCommand(equipmentCode: string, commandType: string, payload: string, createdBy = 'operator') {
    const { data } = await client.post<DeviceCommand>('/api/mes/scada/commands', {
      equipmentCode, commandType, payload, createdBy,
    })
    return data
  },

  async fetchCommands(equipmentCode: string): Promise<DeviceCommand[]> {
    const { data } = await client.get<DeviceCommand[]>(`/api/mes/scada/commands/${equipmentCode}`)
    return data ?? []
  },

  async forwardTrace(traceCode: string): Promise<TraceRecord[]> {
    const { data } = await client.get<TraceRecord[]>('/api/mes/trace/records/forward', {
      params: { traceCode },
    })
    return data ?? []
  },

  async backwardTrace(traceCode: string): Promise<TraceRecord[]> {
    const { data } = await client.get<TraceRecord[]>('/api/mes/trace/records/backward', {
      params: { traceCode },
    })
    return data ?? []
  },

  async getTraceChain(traceCode: string): Promise<TraceChain | null> {
    try {
      const { data } = await client.get<TraceChain>('/api/mes/trace/records/trace-chain', {
        params: { traceCode },
      })
      return data
    } catch {
      return null
    }
  },

  async fetchTraceRecords(params?: {
    productCode?: string
    batchNo?: string
    sn?: string
  }): Promise<TraceRecord[]> {
    const { data } = await client.get<TraceRecord[]>('/api/mes/trace/records', { params })
    return data ?? []
  },
}
