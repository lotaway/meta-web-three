import { describe, it, beforeAll } from '@jest/globals'
import { device, element, by, expect as detoxExpect } from 'detox'

// Minimal real-device smoke test: proves the Detox harness, the built APK and
// the app shell all work together. This is the acceptance bar for re-enabling
// the e2e suite (E2E_SKIP unset, real emulator/device).
describe('app smoke', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true })
  })

  it('launches and renders the home screen', async () => {
    try {
      await detoxExpect(element(by.id('home-page'))).toBeVisible()
    } catch (err) {
      console.error('SMOKE_ASSERT_FAILED', err instanceof Error ? err.message : String(err))
      throw err
    }
  })
})
