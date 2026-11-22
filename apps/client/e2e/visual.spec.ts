import { describe, it, beforeAll, expect } from '@jest/globals'
import { device, expect as detoxExpect, element, by } from 'detox'
import fs from 'fs'
import path from 'path'
import { compareScreenshots } from './visual-compare'

const BASELINE_DIR = path.join(__dirname, 'visual-baselines')
const MAX_DIFF_RATIO = 0.002 // industry-standard pixelmatch default
const SCREEN = 'home'

function baselinePath(): string {
  return path.join(BASELINE_DIR, `${SCREEN}.png`)
}

describe('visual regression (device matrix: android emulator)', () => {
  beforeAll(async () => {
    if (!fs.existsSync(BASELINE_DIR)) fs.mkdirSync(BASELINE_DIR, { recursive: true })
    await device.launchApp({ newInstance: true })
  })

  it('home screen matches the approved baseline', async () => {
    await detoxExpect(element(by.id('home-page'))).toBeVisible()
    const tempPath = await device.takeScreenshot(SCREEN)

    if (process.env.UPDATE_VISUAL_BASELINE === '1') {
      fs.copyFileSync(tempPath, baselinePath())
      return
    }

    const result = compareScreenshots(baselinePath(), tempPath)
    console.log(`visual diff ${SCREEN}: ${(result.ratio * 100).toFixed(3)}% (${result.changedPixels}/${result.totalPixels})`)
    expect(result.ratio).toBeLessThanOrEqual(MAX_DIFF_RATIO)
  })
})
