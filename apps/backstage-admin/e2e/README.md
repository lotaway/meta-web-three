# Visual & accessibility quality gate (opt-in)

The executable structure / budget gate is `tests/quality-gate.spec.ts` (runs with
`yarn test` and in CI). It enforces the industry-standard thresholds centralised
in `quality.config.ts`:

- performance: LCP ≤ 2500ms, CLS ≤ 0.1, FID ≤ 100ms, bundle ≤ 500KB
- accessibility: 0 critical / 0 serious violations
- visual: max pixel-diff ratio 0.2% (pixelmatch-style), per-channel delta 32
- responsive breakpoints: 640 / 1024

## Enabling the screenshot + axe scan

`quality.spec.ts` and `tests/visual-and-a11y.spec.ts` are a ready-to-run Playwright
setup for the visual-consistency and basic-a11y dimensions. To enable:

1. Add dev deps: `@playwright/test`, `@axe-core/playwright`.
2. `npx playwright install --with-deps chromium`.
3. Baseline approval: run `npx playwright test -c e2e/quality.spec.ts --update-snapshots`
   on a reviewed build, commit the generated `*-snapshots/*.png` as the approved
   baseline. Baselines must be approved by a maintainer before merging.
4. Add `yarn test:visual` to CI after the build step (start `yarn preview` first).

The thresholds are read from `quality.config.ts`, so CI and local runs stay in sync.
No baseline is committed yet; the gate is intentionally opt-in to avoid a flaky
first run.
