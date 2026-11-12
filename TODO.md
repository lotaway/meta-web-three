# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

Source: read-only compliance audit of `apps/` against DEVELOPMENT.md, CHECK_LIST.md
and CODE_PINCEPLES (2026-11-12). These are pre-existing compliance debts, not work
done in that session. All items are OPEN; per DEVELOPMENT.md step 4 and CHECK_LIST.md
rule 2 each one records the problem and a suggested fix. Keep them as `[ ]` until a
CHECK_LIST.md verification passes, then remove them.

The final subsection additionally records carry-over items reported in the
2026-11-09 CI cleanup that this rewrite dropped. Each was re-checked against the
repository and GitHub on 2026-11-12; only the items confirmed still open are listed.

### Blocking / security

- [ ] Harden Electron renderer isolation in `apps/digital-twin/system-management`.
  - Problem: `src/main/app-lifecycle.ts:49-50` and `src/main/browser/subtitles-window.ts:45-46` both set `nodeIntegration: true` and `contextIsolation: false`; `app-lifecycle.ts:2,54-55` initializes and enables `@electron/remote`; `src/main/preload.ts:7-37` exposes an unrestricted `send` / `receive` / `ipcSendSync` bridge with no channel allowlist, and `preload.ts:8` swallows invoke errors. A renderer-side XSS therefore reaches Node and IPC directly (RCE).
  - Suggestion: Set `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`, remove `@electron/remote`, keep a preload bridge that exposes only an explicit `IPC_CHANNELS` allowlist, and rethrow or surface invoke errors. Verify local content loads only over `file://` and that every IPC handler validates its input.

- [ ] Add authentication and bind the digital-twin local HTTP server to loopback.
  - Problem: `src/main/http-server.ts:30` sends `Access-Control-Allow-Origin: *`, `:60` calls `listen(this.port)` with no host (all interfaces), and the routes at `:41-56` (screenshot, directory read, video merge, token, TTS synthesize/download/delete) have no authentication.
  - Suggestion: Bind to `127.0.0.1`, replace the wildcard CORS header with an explicit origin or remove it, require a per-session token on every mutating or sensitive route, and validate path inputs in the directory / screenshot controllers.

- [ ] Stop persisting the plaintext login password in `apps/backstage-admin`.
  - Problem: `src/stores/user.ts:25` assigns `loginParam.password` into `userInfo.value.password`, and `:64` enables `persist: true`, so the credential is written to browser storage.
  - Suggestion: Remove the password field from the store state and its persisted payload, keep only the token / user profile, and rotate or clear any already persisted value.

- [ ] Fix auth token storage in `apps/client`.
  - Problem: `src/lib/api/interceptors.ts:8,13,18` read and write `auth_token` via `localStorage`, which is not available in a React Native runtime (no polyfill found in the app) and is not secure storage; a token written there is both a runtime `ReferenceError` risk and plaintext at rest.
  - Suggestion: Use `expo-secure-store` / `SecureStore` (or the platform keystore) through a single token-storage module, add an expiry/refresh path, and add a test that the storage API is available on the target runtime.

### Missing or disabled functionality

- [ ] Replace the throw-instead-of-implementation auth flows in `apps/client`.
  - Problem: `contexts/AuthContext.tsx:237-238`, `:274-275`, `:285-286` consist of a TODO comment plus `throw new Error(...)` in Chinese for phone login, verification code, and forgot-password; the real implementations are commented out. CHECK_RULE.md forbids an exception in place of an implementation.
  - Suggestion: Wire the three flows to the generated API endpoints, keep English error messages, and cover success / failure / rate-limit cases with unit tests; remove the dead commented code.

- [ ] Re-enable the `apps/client` e2e suite and repair its config.
  - Problem: `e2e/setup.ts:28` assigns `global.describe = global.describe.skip`, which permanently disables every detox spec; `e2e/jest.config.js` is a bare object literal with no `module.exports`, and `require()` fails with `Unexpected token ':'`.
  - Suggestion: Delete the global skip, export the config with `module.exports`, run at least one real spec in CI, and guard any environment-specific skip behind an explicit env flag with a log.

### Automated quality gates

- [ ] Implement the FRONTEND_PRICEPLES visual and five-dimension quality gates.
  - Problem: Section 6 (`:49-50`) requires an executable visual quality detection mechanism with quantified diff criteria; section 9 (`:68-69`) requires automated coverage of structural correctness, responsive layout, visual consistency, performance metrics and basic accessibility. No visual / screenshot / a11y / performance test file exists anywhere under `apps/`. Existing performance assertions only check `not.toThrow()` (e.g. `AlertFlow.test.tsx:695`, `RealTimeDataDisplay.test.tsx:226`).
  - Suggestion: Add a runnable screenshot-diff / visual regression job with an explicit pixel or perceptual threshold, plus a11y and performance budgets wired into CI; decide the tool per app (Playwright for the two web frontends, a device-matrix snapshot runner for the Expo app).

- [ ] Bring per-app test coverage up to the project's test framework requirements.
  - Problem: `backstage-admin` has 1 test file (`tests/env.spec.ts`) for 93 handwritten `src/apis/` modules; `apps/client` has 9 test files, all in the `lib` layer, with no component / screen coverage and a disabled e2e suite; `digital-twin/system-management` has 7 unit test files plus `playwright.config.ts` and one `src/renderer/e2e/scene.spec.ts`, with no responsive / visual / performance coverage.
  - Suggestion: Add tests for API modules and critical pages/screens in each app, make the e2e specs runnable in CI, and assert behavior rather than absence of exceptions.

### Code principles and frontend standards

- [ ] Clean up code-principle violations across all three clients.
  - Problem: DEVELOPMENT.md `:82-85` requires English-only code text (except i18n), functions under 20 lines, no swallowed exceptions, no hidden shared state and guard clauses. A static scan of `src/` found Chinese text on 5751 lines in backstage-admin, 891 in client and 509 in digital-twin (i18n strings must be separated from comments/logs during the fix); catch blocks number 581 / 15 / 65 respectively, and several only log and return; the audit also reported a roughly 227-line function in the digital-twin controllers (re-measure before fixing).
  - Suggestion: Work app by app: translate comments, logs and error strings (keep i18n resources as the only Chinese text), turn logged-and-ignored catches into explicit error states or rethrows, and split functions over 20 lines. Re-run each app's typecheck and tests after each pass.

- [ ] Close the responsive and accessibility gaps in all three clients.
  - Problem: Static scan of `src/`: `@media` appears in 4 backstage-admin files and 0 files in the other two apps; `aria-*` hits are 1 / 0 / 21 and `useWindowDimensions` / `Dimensions` hits are 0 / 1 / 0; `apps/backstage-admin/src/styles/index.scss:32,37` remove focus outlines with `outline: none`; `apps/client` has no accessibility props or ErrorBoundary and locks orientation to portrait (`app.config.ts:9`, `android/app/src/main/AndroidManifest.xml:19`); `digital-twin` has no media queries.
  - Suggestion: Add breakpoints and flexible layouts per app, restore visible focus states, add semantic labels / `aria-*` or `accessibilityLabel` / `accessibilityRole`, and add an ErrorBoundary plus user-visible degradation for load failures (FRONTEND_PRICEPLES sections 7 and 8).

### API client chain and documentation

- [ ] Reconcile the `apps/backstage-admin` API client with DEVELOPMENT.md section 3.
  - Problem: `package.json:18` defines `generate:api` and `tools/generate-api.cjs` calls the shared `apps/tools/OpenApiToTS.js` whose default output is `src/generated/api`, but that directory does not exist; the app instead calls 93 handwritten modules under `src/apis/`.
  - Suggestion: Either run the generator and migrate consumers to the generated client, or document the deliberate deviation (including why the handwritten modules are kept and how they stay in sync) in DEVELOPMENT.md and the app README.

- [ ] Align DEVELOPMENT.md section 3 with the actual env and generator contracts.
  - Problem: `DEVELOPMENT.md:49-58` requires `.env.development` to define `NEXT_PUBLIC_BACK_API_HOST` / `NEXT_PUBLIC_BACK_API_DOC_HOST` and `src/generated/api/` output for both apps. In reality `backstage-admin/.env.development` only sets `VITE_BLOCKCHAIN_ENABLED=true` and its generator reads `.env,.env.local` with `VITE_BASE_SERVER_URL` / `VITE_API_DOC_HOST`; `apps/client` keeps the `NEXT_PUBLIC_*` variables in `.env` and has 1536 generated files; `digital-twin` correctly has no `generate:api` and 5 handwritten type files. The doc and both implementations disagree.
  - Suggestion: Pick one env-file / variable convention, update the three apps and `.env.example` files accordingly, and rewrite section 3 to describe the real per-app behavior, including the digital-twin exception.

### CI and release verification carry-over

Source: items reported as outstanding on 2026-11-09 that were lost when TODO.md was
rewritten by the compliance audit. Re-verified on 2026-11-12. Partially resolved work
is noted inline so it is not re-done; all four items below are still OPEN.

- [ ] Revoke the exposed classic PAT and purge it from git history.
  - Problem: Commit `3c07b882` (2026-05-26) stores a plaintext `ghp_…` token in `package.json` `repository.url`; the value is still readable from the object database even though the current `package.json`, `.git/config` and `remote.origin.url` are clean and `lefthook.yml:23-28` now rejects new tokens in added lines. The historical token already returns HTTP 401, but it was never revoked or purged. `~/.git-credentials` still holds a classic PAT with `repo, workflow` scopes that authenticates successfully (`GET /user` → 200, login `lotaway`).
  - Suggestion: Delete/revoke both classic PATs in GitHub Settings, replace them with a fine-grained PAT or the OS credential manager, and purge the historical value (`git filter-repo` / BFG) with a coordinated force-push and post-purge rotation. Add a full-history secret scan in CI so this cannot recur; if a history rewrite is rejected, record the accepted risk explicitly.

- [ ] Point the performance pipeline at a real staging target.
  - Problem: `GET /repos/lotaway/meta-web-three/actions/variables` returns `total_count: 0`, so `vars.PERF_BASE_URL` is unset. `.github/workflows/performance-test.yml:107,116,124-141` therefore forces `use_mock = True` on every scheduled run and falls back to the built-in `127.0.0.1:18087` mock on manual runs; runs #163–#165 are green but never touched a real service. The `environment` input (`:16`) is documented as "Report label only", so a run labelled `staging` can silently report mock numbers, and the `schedule` branch hardcodes `use_mock = True` at `:127` — configuring the variable alone will not fix scheduled runs.
  - Suggestion: Set the `PERF_BASE_URL` repository/environment variable to a reachable staging gateway, remove the unconditional `use_mock = True` from the schedule branch (or gate it behind an explicit `scheduled-smoke` opt-in), and fail or clearly label the report whenever the mock is used so mock numbers can never be read as staging results. Re-run a `workflow_dispatch` against the real target and confirm the target in the step output before removing this item.

- [ ] Add reproducible installs and an Electron runtime smoke check for `apps/digital-twin/system-management`.
  - Problem: `d10f4015` added `.github/workflows/system-management-check.yml` (run #1 success), fixed the build context / Dockerfile / compose paths and added `npm run build:native`, `npm run typecheck` and `npm run test:run` in CI, which resolves the earlier "no CI builds it" gap. Still open: no lockfile exists anywhere under `apps/` (the workflow comments at `:34-35` disable the package-manager cache for exactly this reason), `node_modules` is not installed for this app, `docker/Dockerfile:20-24` and the workflow `:49` both use unpinned `npm install --legacy-peer-deps`, neither runs `npm run rebuild:sqlite3` (the app declares `better-sqlite3 ^11.0.0`), and neither launches Electron (`ELECTRON_SKIP_BINARY_DOWNLOAD=1` at `docker/Dockerfile:4` and workflow `:48`). Native ABI regressions and the two insecure `BrowserWindow` configs tracked in "Harden Electron renderer isolation" above are therefore never exercised at runtime. A mirror gap on 2026-11-09 (`npmmirror` missing `@typescript-eslint/parser@8.72.2`) already broke installs; the mirror has since synced, but installs remain unpinned.
  - Suggestion: Commit a lockfile for this app (then for the other `apps/*` packages), switch CI and Docker back to `npm ci`, pin the registry via `.npmrc`, add a CI step for `npm run rebuild:sqlite3` plus a headless Electron launch/smoke (or document the required manual device check), and record the result of a real-device run of both windows before removing this item.

- [ ] Make `apps/backstage-admin` env-default tests hermetic.
  - Problem: `tests/env.spec.ts:13-16` asserts `BASE_SERVER_URL === DEFAULT_BASE_SERVER_URL` while `src/utils/env.ts:27-30` reads `import.meta.env.VITE_BASE_SERVER_URL`, which Vite populates from the developer's gitignored `.env`. With the local `.env` (`VITE_BASE_SERVER_URL=http://192.168.1.110:10081`) present, `npx vitest run tests/env.spec.ts` fails 1/5 (`expected 'http://192.168.1.110:10081' to be 'http://localhost:10081'`); it only passes when the variable is cleared or in CI where no `.env` exists. `vite.config.ts:153-156` sets no test-only env.
  - Suggestion: Isolate the test from ambient `.env` (refactor `env.ts` to expose a pure `resolveEnv(vars)` factory tested with explicit inputs, or stub `import.meta.env` before a dynamic `import()` with `vi.resetModules()`), keep `.env.example` defaults documented, and add a CI case that runs the suite with a non-default `.env` to prove hermeticity.

## Completed Tasks (archived)

English notes only. Each item was verified with CHECK_LIST.md before archival.
Per DEVELOPMENT.md step 4, passed tasks are removed from the actionable list and
recorded here as checked items.

- [x] Clear completed and unverifiable tasks from TODO.md. Commit: `57d1db9a`.
- [x] Generate native types before typecheck in CI and Docker; fix the digital-twin build context, Dockerfile, compose paths and add the check workflow. Commit: `d10f4015`.
- [x] Fix the pre-commit guards so they no longer fail on Windows; verify the hooks with staged probes. Commit: `ccb508d1`.
