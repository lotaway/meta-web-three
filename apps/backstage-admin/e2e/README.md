# Visual & accessibility quality gate

The five-dimension budgets are centralised in `quality.config.ts` and consumed by
both the vitest gate (`tests/quality-gate.spec.ts`, runs in `yarn test`) and the
Playwright visual/a11y scan in this directory.

Thresholds (industry-standard defaults):
- performance: LCP ≤ 2500ms, CLS ≤ 0.1, FID ≤ 100ms, bundle ≤ 500KB
- accessibility: 0 critical / 0 serious axe violations
- visual: max pixel-diff ratio 0.2%, per-channel delta 32
- responsive breakpoints: 640 / 1024

## Running locally

```bash
npm run build-only
npx vite preview --host 127.0.0.1 --port 4173 &
npm run test:visual            # compare against committed baseline
npm run test:visual:update     # regenerate baseline after an approved UI change
```

The config uses `channel: 'chrome'` so an installed Chrome/Edge is used instead
of downloading a 200MB bundled browser. Override with `PLAYWRIGHT_CHANNEL` or
set it to an empty value to fall back to bundled chromium.

## Baseline approval

`tests/visual-and-a11y.spec.ts-snapshots/login-win32.png` is the committed,
approved baseline for the login page (the only route reachable without a backend
or auth). Any UI change that alters it must be reviewed, regenerated with
`test:visual:update`, and committed in the same change.

Evidence (2026-11-21):
- passing run: 1 passed (baseline match + 0 critical/serious axe violations)
- deliberate failure: baseline corrupted → `toHaveScreenshot` failed (1 failed),
  then baseline restored → passing again
- the gate surfaced two real serious a11y issues on the login page, both fixed:
  empty `<html lang>` and insufficient white-on-`#409eff` contrast (2.78:1),
  resolved by setting `lang="zh-CN"` and darkening the primary color to `#2d6fcc`
  (~4.8:1).
