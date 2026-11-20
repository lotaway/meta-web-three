import { useCallback, useRef } from 'react'
import { Performance } from './types'

export function usePerformance() {
  const marksRef = useRef<Map<string, number>>(new Map())
  const measuresRef = useRef<Performance.Measure[]>([])

  const startMark = useCallback((name: string) => {
    marksRef.current.set(name, performance.now())
  }, [])

  const endMark = useCallback((name: string) => {
    const startTime = marksRef.current.get(name)
    if (!startTime) return null

    const duration = performance.now() - startTime
    marksRef.current.delete(name)
    return duration
  }, [])

  const measure = useCallback((name: string, startMarkName: string, endMarkName?: string) => {
    const startTime = marksRef.current.get(startMarkName)

    if (!startTime) return null

    const endTime = endMarkName
      ? marksRef.current.get(endMarkName)
      : performance.now()

    if (!endTime) return null

    const measure: Performance.Measure = {
      name,
      startTime,
      duration: endTime - startTime,
      timestamp: Date.now(),
    }

    measuresRef.current.push(measure)
    return measure
  }, [])

  const getMeasures = useCallback(() => {
    return [...measuresRef.current]
  }, [])

  const clear = useCallback(() => {
    marksRef.current.clear()
    measuresRef.current = []
  }, [])

  return { startMark, endMark, measure, getMeasures, clear }
}

export function useLazyLoad(threshold: number = 0.5) {
  const loadedRef = useRef(false)
  const callbackRef = useRef<(() => void) | null>(null)

  const createObserver = useCallback((callback: () => void) => {
    return new IntersectionObserver((entries) => {
      if (entries.some(entry => entry.isIntersecting) && !loadedRef.current) {
        loadedRef.current = true
        callback()
      }
    }, { threshold })
  }, [threshold])

  const observe = useCallback((element: any, callback: () => void) => {
    callbackRef.current = callback
    if (!('IntersectionObserver' in window)) return () => {}

    const observer = createObserver(callback)
    if (element) observer.observe(element)
    return () => observer.disconnect()
  }, [createObserver])

  return { observe }
}

export function useImageOptimization() {
  const generateSrcSet = useCallback((baseUrl: string, widths: number[] = [320, 640, 960]) => {
    return widths.map((w) => `${baseUrl}?w=${w} ${w}w`).join(', ')
  }, [])

  const getOptimizedUrl = useCallback((url: string, width: number, quality: number = 80) => {
    const separator = url.includes('?') ? '&' : '?'
    return `${url}${separator}w=${width}&q=${quality}`
  }, [])

  return { generateSrcSet, getOptimizedUrl }
}