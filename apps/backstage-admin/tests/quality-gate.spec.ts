import { describe, expect, it } from 'vitest'
import { qualityConfig } from '../quality.config'

describe('quality gate budgets', () => {
  it('uses industry-standard performance thresholds', () => {
    expect(qualityConfig.performance.lcpMs).toBeLessThanOrEqual(2500)
    expect(qualityConfig.performance.cls).toBeLessThanOrEqual(0.1)
    expect(qualityConfig.performance.fidMs).toBeLessThanOrEqual(100)
  })

  it('has zero tolerance for critical and serious a11y violations', () => {
    expect(qualityConfig.accessibility.maxCriticalViolations).toBe(0)
    expect(qualityConfig.accessibility.maxSeriousViolations).toBe(0)
  })

  it('keeps the visual pixel-diff threshold below one percent', () => {
    expect(qualityConfig.visual.maxPixelDiffRatio).toBeGreaterThan(0)
    expect(qualityConfig.visual.maxPixelDiffRatio).toBeLessThanOrEqual(0.01)
  })

  it('covers the responsive breakpoints used by the app', () => {
    expect(qualityConfig.responsive.breakpoints).toEqual([640, 1024])
  })
})
