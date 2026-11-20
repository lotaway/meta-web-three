import { useCallback, useRef } from 'react'
import { Performance } from './types'

type MarksRef = { current: Map<string, number> }
type MeasuresRef = { current: Performance.Measure[] }

function createMeasure(name: string, startTime: number, endTime: number): Performance.Measure {
  return {
    name,
    startTime,
    duration: endTime - startTime,
    timestamp: Date.now(),
  }
}

function useMarks() {
  const marksRef = useRef<Map<string, number>>(new Map())
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
  return { marksRef, startMark, endMark }
}

function useMarkMeasure(marksRef: MarksRef, measuresRef: MeasuresRef) {
  return useCallback((name: string, startMarkName: string, endMarkName?: string) => {
    const startTime = marksRef.current.get(startMarkName)
    if (!startTime) return null
    const endTime = endMarkName
      ? marksRef.current.get(endMarkName)
      : performance.now()
    if (!endTime) return null
    const created = createMeasure(name, startTime, endTime)
    measuresRef.current.push(created)
    return created
  }, [])
}

function useMeasures(marksRef: MarksRef) {
  const measuresRef = useRef<Performance.Measure[]>([])
  const measure = useMarkMeasure(marksRef, measuresRef)
  const getMeasures = useCallback(() => [...measuresRef.current], [])
  const clear = useCallback(() => {
    marksRef.current.clear()
    measuresRef.current = []
  }, [])
  return { measure, getMeasures, clear }
}

export function usePerformance() {
  const { marksRef, startMark, endMark } = useMarks()
  const { measure, getMeasures, clear } = useMeasures(marksRef)
  return { startMark, endMark, measure, getMeasures, clear }
}

function createLazyObserver(
  callback: () => void,
  loadedRef: { current: boolean },
  threshold: number
) {
  return new IntersectionObserver((entries) => {
    if (entries.some(entry => entry.isIntersecting) && !loadedRef.current) {
      loadedRef.current = true
      callback()
    }
  }, { threshold })
}

export function useLazyLoad(threshold: number = 0.5) {
  const loadedRef = useRef(false)
  const callbackRef = useRef<(() => void) | null>(null)
  const createObserver = useCallback((callback: () => void) => {
    return createLazyObserver(callback, loadedRef, threshold)
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
