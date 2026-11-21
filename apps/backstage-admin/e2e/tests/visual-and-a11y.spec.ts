import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

// Baselines target pages reachable without a backend or auth so the gate is
// deterministic in CI. Authenticated pages can be added once a seeded env exists.
const routes = [
  { name: 'login', path: '/login', snapshot: 'login.png' },
]

for (const route of routes) {
  test(`${route.name}: visual baseline and no serious a11y violations`, async ({ page }) => {
    await page.goto(route.path)
    await expect(page.locator('body')).toBeVisible()
    await expect(page).toHaveScreenshot(route.snapshot, { fullPage: true })

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze()
    const critical = results.violations.filter((v) => v.impact === 'critical')
    const serious = results.violations.filter((v) => v.impact === 'serious')
    expect(critical).toHaveLength(0)
    expect(serious).toHaveLength(0)
  })
}
