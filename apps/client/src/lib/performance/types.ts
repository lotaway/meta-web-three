export namespace Performance {
  export interface Measure {
    name: string
    startTime: number
    duration: number
    timestamp: number
  }

  export interface MemoryUsage {
    used: number
    total: number
    percentage: number
  }
}

export type PerformanceObserverCallback = (entry: PerformanceEntry) => void

export interface PerformanceOptions {
  reportOnConsole?: boolean
  threshold?: number
}