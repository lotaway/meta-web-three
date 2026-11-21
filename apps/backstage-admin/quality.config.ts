/**
 * Industry-standard quality budgets for the five dimensions required by
 * FRONTEND_PRICEPLES section 9. Values are intentionally conservative and
 * centralised so CI and local runs use the same thresholds.
 */
export const qualityConfig = {
  structure: {
    maxFunctionLines: 20,
    maxFileLines: 500,
  },
  responsive: {
    breakpoints: [640, 1024],
    // Every breakpoint must be reachable without horizontal overflow.
    allowedHorizontalOverflowPx: 0,
  },
  visual: {
    // pixelmatch-style ratio; 0.2% is the common default for screenshot diffs.
    maxPixelDiffRatio: 0.002,
    maxPerChannelDelta: 32,
  },
  performance: {
    // Lighthouse "good" thresholds.
    lcpMs: 2500,
    cls: 0.1,
    fidMs: 100,
    maxBundleBytes: 500_000,
  },
  accessibility: {
    // Zero tolerance for critical/serious axe rules; moderate is tracked.
    maxCriticalViolations: 0,
    maxSeriousViolations: 0,
    requiredRoles: ['button', 'link', 'textbox', 'alert'],
  },
} as const

export type QualityConfig = typeof qualityConfig
