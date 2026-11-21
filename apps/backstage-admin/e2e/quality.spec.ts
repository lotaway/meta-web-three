import { defineConfig } from '@playwright/test'
import { qualityConfig } from '../quality.config'

export default defineConfig({
  testDir: './tests',
  use: {
    baseURL: process.env.QUALITY_BASE_URL ?? 'http://localhost:4173',
    viewport: { width: qualityConfig.responsive.breakpoints[1], height: 900 },
    // Use an installed browser so CI/local runs need no 200MB browser download.
    // Set PLAYWRIGHT_CHANNEL='' to fall back to bundled chromium.
    channel: process.env.PLAYWRIGHT_CHANNEL === undefined ? 'chrome' : (process.env.PLAYWRIGHT_CHANNEL || undefined),
  },
  expect: {
    // Screenshot diffs use toHaveScreenshot with a pixel-ratio derived threshold.
    toHaveScreenshot: {
      maxDiffPixelRatio: qualityConfig.visual.maxPixelDiffRatio,
      threshold: qualityConfig.visual.maxPerChannelDelta / 255,
    },
  },
})
