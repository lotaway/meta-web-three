import { useCallback, useRef } from 'react'

export function useCallbackRef<T extends (...args: any[]) => any>(callback: T): T {
  const callbackRef = useRef(callback)
  callbackRef.current = callback

  const memoizedCallback = useCallback(
    (...args: Parameters<T>) => callbackRef.current(...args),
    []
  ) as T

  return memoizedCallback
}

export function useStableCallback<T extends (...args: any[]) => any>(callback: T): T {
  const ref = useRef(callback)
  ref.current = callback

  return useCallback((...args: Parameters<T>) => ref.current(...args), []) as T
}

export interface MemoryInfo {
  jsHeapSizeLimit: number
  totalJSHeapSize: number
  usedJSHeapSize: number
}

export function getMemoryInfo(): MemoryInfo | null {
  if (typeof window === 'undefined' || !(window as any).performance?.memory) {
    return null
  }

  const memory = (window as any).performance.memory
  return {
    jsHeapSizeLimit: memory.jsHeapSizeLimit,
    totalJSHeapSize: memory.totalJSHeapSize,
    usedJSHeapSize: memory.usedJSHeapSize,
  }
}

export function useMemoryWarning(threshold: number = 0.9) {
  const checkMemory = useCallback(() => {
    const info = getMemoryInfo()
    if (!info) return false

    const usage = info.usedJSHeapSize / info.jsHeapSizeLimit
    return usage > threshold
  }, [threshold])

  return { checkMemory, getMemoryInfo }
}

export function createMemoryMonitor(onWarning?: (usage: number) => void) {
  let intervalId: NodeJS.Timeout | null = null

  const start = (threshold: number = 0.9, interval: number = 5000) => {
    if (typeof window === 'undefined') return

    intervalId = setInterval(() => {
      const info = getMemoryInfo()
      if (!info) return

      const usage = info.usedJSHeapSize / info.jsHeapSizeLimit
      if (usage > threshold && onWarning) {
        onWarning(usage)
      }
    }, interval)
  }

  const stop = () => {
    if (intervalId) {
      clearInterval(intervalId)
      intervalId = null
    }
  }

  return { start, stop }
}

export function useRefreshControlProps(refreshing: boolean) {
  return {
    refreshing,
    onRefresh: undefined,
    progressViewOffset: 0,
  }
}