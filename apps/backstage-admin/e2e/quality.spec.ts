import { defineConfig } from '@playwright/test'
import { qualityConfig } from '../quality.config'

export default defineConfig({
  testDir: './tests',
  use: {
    baseURL: process.env.QUALITY_BASE_URL ?? 'http://localhost:10081',
    viewport: { width: qualityConfig.responsive.breakpoints[1], height: 900 },
  },
  expect: {
    // Screenshot diffs use toHaveScreenshot with a pixel-ratio derived threshold.
    toHaveScreenshot: {
      maxDiffPixelRatio: qualityConfig.visual.maxPixelDiffRatio,
      threshold: qualityConfig.visual.maxPerChannelDelta / 255,
    },
  },
})
