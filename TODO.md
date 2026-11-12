# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

Source: read-only compliance audit of `apps/` against DEVELOPMENT.md, CHECK_LIST.md
and CODE_PINCEPLES (2026-11-12). These are pre-existing compliance debts, not work
done in that session. All items below are OPEN; per DEVELOPMENT.md step 4 and
CHECK_LIST.md rule 2 each one records the problem and a suggested fix.

Regrouped on 2026-11-12 by **what is needed to close each item**, replacing the
earlier topic-based sections. Four groups: local-only work, decisions that must
be made by a person, environment / permission / external-service requirements,
and items needing both. Every item now carries a `Gate:` line with its closing
conditions, plus `Decision:` and/or `Requires:` lines naming the exact missing
input or resource. Keep an item as `[ ]` until a CHECK_LIST.md verification
passes, then move it to the archive below. Original Problem / Suggestion text
is preserved so the audit evidence is not lost.

The three PAT / performance / reproducible-install items are carry-overs
reported in the 2026-11-09 CI cleanup that the compliance-audit rewrite
dropped; they were re-verified against the repository and GitHub on 2026-11-12.
The fourth carry-over (env-default test hermeticity) was fixed on 2026-11-12
and moved to the archive.

### 1. Local-only (no decision or external resource needed)

These can be implemented and verified on this host without any input or
additional service.

- [ ] Stop persisting the plaintext login password in `apps/backstage-admin`.
  - Problem: `src/stores/user.ts:25` assigns `loginParam.password` into `userInfo.value.password`, and `:64` enables `persist: true`, so the credential is written to browser storage.
  - Suggestion: Remove the password field from the store state and its persisted payload, keep only the token / user profile, and rotate or clear any already persisted value.
  - Gate: Remove the field and its persisted payload, add a migration that deletes any previously stored value, add a unit test, and pass `vue-tsc --build` plus vitest. No external input is needed; the default is to keep the existing session and only clear the stored credential, without forcing a re-login.

### 2. Decision-gated (needs a human choice)

The code work is local, but a product / contract / acceptance decision must be
made first or the fix risks being reverted.

- [ ] Harden Electron renderer isolation in `apps/digital-twin/system-management`.
  - Problem: `src/main/app-lifecycle.ts:49-50` and `src/main/browser/subtitles-window.ts:45-46` both set `nodeIntegration: true` and `contextIsolation: false`; `app-lifecycle.ts:2,54-55` initializes and enables `@electron/remote`; `src/main/preload.ts:7-37` exposes an unrestricted `send` / `receive` / `ipcSendSync` bridge with no channel allowlist, and `preload.ts:8` swallows invoke errors. A renderer-side XSS therefore reaches Node and IPC directly (RCE).
  - Suggestion: Set `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`, remove `@electron/remote`, keep a preload bridge that exposes only an explicit `IPC_CHANNELS` allowlist, and rethrow or surface invoke errors. Verify local content loads only over `file://` and that every IPC handler validates its input.
  - Decision: Confirm the renderer-side IPC call sites before shrinking the bridge; an allowlist is a breaking contract change. Choose whether to keep a temporary compatibility shim.
  - Gate: Apply the BrowserWindow / preload / remote changes, rethrow invoke errors, enumerate and validate every IPC handler's input, then launch both BrowserWindows and record the result. A GUI session is available on this Windows host; a CI headless run would additionally need xvfb and the Electron binary.

- [ ] Add authentication and bind the digital-twin local HTTP server to loopback.
  - Problem: `src/main/http-server.ts:30` sends `Access-Control-Allow-Origin: *`, `:60` calls `listen(this.port)` with no host (all interfaces), and the routes at `:41-56` (screenshot, directory read, video merge, token, TTS synthesize/download/delete) have no authentication.
  - Suggestion: Bind to `127.0.0.1`, replace the wildcard CORS header with an explicit origin or remove it, require a per-session token on every mutating or sensitive route, and validate path inputs in the directory / screenshot controllers.
  - Decision: Choose the per-session token scheme and confirm the callers of this local HTTP server (external digital-twin processes may rely on the current unauthenticated routes). Adding auth is a breaking interface change.
  - Gate: Bind to `127.0.0.1`, remove the wildcard CORS header, enforce the token on every sensitive route, validate path inputs, and confirm the known callers still work.

- [ ] Bring per-app test coverage up to the project's test framework requirements.
  - Problem: `backstage-admin` has 1 test file (`tests/env.spec.ts`) for 93 handwritten `src/apis/` modules; `apps/client` has 9 test files, all in the `lib` layer, with no component / screen coverage and a disabled e2e suite; `digital-twin/system-management` has 7 unit test files plus `playwright.config.ts` and one `src/renderer/e2e/scene.spec.ts`, with no responsive / visual / performance coverage.
  - Suggestion: Add tests for API modules and critical pages/screens in each app, make the e2e specs runnable in CI, and assert behavior rather than absence of exceptions.
  - Decision: Fix the acceptance bar. DEVELOPMENT.md only says "add tests matching the project's test framework"; if a numeric coverage threshold is required, the target percentage must be chosen first. Behavioural assertions can be written without it.
  - Gate: Add API-module and critical page/screen tests in all three apps that assert behaviour rather than `not.toThrow()`, make the e2e specs runnable in CI, and re-run each app's typecheck and tests.

- [ ] Clean up code-principle violations across all three clients.
  - Problem: DEVELOPMENT.md `:82-85` requires English-only code text (except i18n), functions under 20 lines, no swallowed exceptions, no hidden shared state and guard clauses. A static scan of `src/` found Chinese text on 5751 lines in backstage-admin, 891 in client and 509 in digital-twin (i18n strings must be separated from comments/logs during the fix); catch blocks number 581 / 15 / 65 respectively, and several only log and return; the audit also reported a roughly 227-line function in the digital-twin controllers (re-measure before fixing).
  - Suggestion: Work app by app: translate comments, logs and error strings (keep i18n resources as the only Chinese text), turn logged-and-ignored catches into explicit error states or rethrows, and split functions over 20 lines. Re-run each app's typecheck and tests after each pass.
  - Decision: Define which Chinese strings count as i18n resources (only those may stay). The 5751 / 891 / 509 line counts mix i18n text, comments, logs and error strings, so a path-based classification rule must be agreed before a bulk rewrite.
  - Gate: Work app by app, translate comments / logs / errors to English, turn logged-and-ignored catches into explicit error states or rethrows, split functions over 20 lines (re-measure the reported ~227-line controller first), and pass typecheck plus tests after each pass.

- [ ] Close the responsive and accessibility gaps in all three clients.
  - Problem: Static scan of `src/`: `@media` appears in 4 backstage-admin files and 0 files in the other two apps; `aria-*` hits are 1 / 0 / 21 and `useWindowDimensions` / `Dimensions` hits are 0 / 1 / 0; `apps/backstage-admin/src/styles/index.scss:32,37` remove focus outlines with `outline: none`; `apps/client` has no accessibility props or ErrorBoundary and locks orientation to portrait (`app.config.ts:9`, `android/app/src/main/AndroidManifest.xml:19`); `digital-twin` has no media queries.
  - Suggestion: Add breakpoints and flexible layouts per app, restore visible focus states, add semantic labels / `aria-*` or `accessibilityLabel` / `accessibilityRole`, and add an ErrorBoundary plus user-visible degradation for load failures (FRONTEND_PRICEPLES sections 7 and 8).
  - Decision: Provide the design baseline / breakpoint values (no design source is in the repository). Without it the default is a conventional breakpoint set submitted as a reviewable change.
  - Gate: Add breakpoints and flexible layouts in all three apps, restore visible focus states, add semantic labels / `aria-*` / `accessibilityLabel` / `accessibilityRole`, add an ErrorBoundary with user-visible degradation, and verify with each app's checks.

- [ ] Reconcile the `apps/backstage-admin` API client with DEVELOPMENT.md section 3.
  - Problem: `package.json:18` defines `generate:api` and `tools/generate-api.cjs` calls the shared `apps/tools/OpenApiToTS.js` whose default output is `src/generated/api`, but that directory does not exist; the app instead calls 93 handwritten modules under `src/apis/`.
  - Suggestion: Either run the generator and migrate consumers to the generated client, or document the deliberate deviation (including why the handwritten modules are kept and how they stay in sync) in DEVELOPMENT.md and the app README.
  - Decision: Choose one of the two routes in the suggestion: run the generator and migrate the 93 handwritten modules, or document the deliberate deviation. Decide together with "Align DEVELOPMENT.md section 3" below, otherwise the two changes conflict.
  - Requires (generator route only): a reachable backend OpenAPI document and the generator toolchain.
  - Gate: Either migrate the consumers and produce `src/generated/api/`, or land the documented deviation in DEVELOPMENT.md and the app README with an explicit sync mechanism.

- [ ] Align DEVELOPMENT.md section 3 with the actual env and generator contracts.
  - Problem: `DEVELOPMENT.md:49-58` requires `.env.development` to define `NEXT_PUBLIC_BACK_API_HOST` / `NEXT_PUBLIC_BACK_API_DOC_HOST` and `src/generated/api/` output for both apps. In reality `backstage-admin/.env.development` only sets `VITE_BLOCKCHAIN_ENABLED=true` and its generator reads `.env,.env.local` with `VITE_BASE_SERVER_URL` / `VITE_API_DOC_HOST`; `apps/client` keeps the `NEXT_PUBLIC_*` variables in `.env` and has 1536 generated files; `digital-twin` correctly has no `generate:api` and 5 handwritten type files. The doc and both implementations disagree.
  - Suggestion: Pick one env-file / variable convention, update the three apps and `.env.example` files accordingly, and rewrite section 3 to describe the real per-app behavior, including the digital-twin exception.
  - Decision: Pick one env-file / variable convention across the three apps (`NEXT_PUBLIC_*` vs `VITE_*`, `.env.development` vs `.env`). Decide together with "Reconcile the backstage-admin API client" above.
  - Gate: Update the three apps and the `.env.example` files, then rewrite DEVELOPMENT.md section 3 to describe the real per-app behaviour including the digital-twin exception.

### 3. Environment / permission-gated (needs a device, account or service)

The remaining code work is small and local, but closing the item requires a
runtime that is not available in a plain checkout.

- [ ] Re-enable the `apps/client` e2e suite and repair its config.
  - Problem: `e2e/setup.ts:28` assigned `global.describe = global.describe.skip`, which permanently disabled every detox spec; `e2e/jest.config.js` was a bare object literal with no `module.exports`, and `require()` failed with `Unexpected token ':'`. Both code defects were fixed on 2026-11-12: the skip now requires an explicit `E2E_SKIP=1`, the config exports a module and `setupFilesAfterEnv` points at `<rootDir>/setup.ts`. Still open: no detox spec has been run on a simulator or device, and no CI job runs the suite.
  - Additional findings (2026-11-12): four pre-existing defects still block any device run and were not recorded above. `e2e/setup.ts` imports `detox-adapter-jest` and `detox-matchers`, neither declared in `package.json` (they appear nowhere else in the repo); `e2e/jest.config.js:5` reporters include the undeclared `detox-runner-reporter`; `e2e/detox.config.js` uses the pre-Detox-20 `composition` / `session` shape, sits outside the default `.detoxrc.js` path, and line 18 contains a stray non-ASCII key `油脂: 'keep'`; `e2e/app.spec.ts:3-13` imports `../pages` while the page objects live at `e2e/pages/index.ts`. These can be fixed locally.
  - Suggestion: Run at least one real spec with `E2E_SKIP` unset (CI or a documented manual device check) and record the result before removing this item. Do not reintroduce a global skip.
  - Requires: A simulator or connected device - Android SDK plus an emulator image (possible on this Windows host), or macOS + Xcode for the configured `iPhone 15 Pro` / iOS 17.0 device. Keeping the suite enabled long-term needs a CI job with the same runtime or a documented manual device check.
  - Gate: Declare the missing dependencies, rewrite the config for Detox 20, fix the spec import, then run at least one real spec with `E2E_SKIP` unset and record the result.

### 4. Decision + environment gated (needs both)

Each item needs an explicit approval or contract decision **and** an external
resource (device build, account permission, reachable service or CI runtime).

- [ ] Fix auth token storage in `apps/client`.
  - Problem: `src/lib/api/interceptors.ts:8,13,18` read and write `auth_token` via `localStorage`, which is not available in a React Native runtime (no polyfill found in the app) and is not secure storage; a token written there is both a runtime `ReferenceError` risk and plaintext at rest.
  - Suggestion: Use `expo-secure-store` / `SecureStore` (or the platform keystore) through a single token-storage module, add an expiry/refresh path, and add a test that the storage API is available on the target runtime.
  - Decision: Approve adding the `expo-secure-store` native dependency and choose the expiry / refresh policy.
  - Requires: A rebuilt Expo dev client (`expo run:android` or an EAS dev build). SecureStore is a native module and cannot be exercised in Expo Go, and the target-runtime availability test needs such a build.
  - Gate: Route all token reads / writes through one SecureStore-backed module, add expiry / refresh handling, add a test asserting the storage API is available on the target runtime, and verify on a built dev client.

- [ ] Replace the throw-instead-of-implementation auth flows in `apps/client`.
  - Problem: `contexts/AuthContext.tsx:237-238`, `:274-275`, `:285-286` consist of a TODO comment plus `throw new Error(...)` in Chinese for phone login, verification code, and forgot-password; the real implementations are commented out. CHECK_RULE.md forbids an exception in place of an implementation.
  - Suggestion: Wire the three flows to the generated API endpoints, keep English error messages, and cover success / failure / rate-limit cases with unit tests; remove the dead commented code.
  - Decision: Provide the three endpoint contracts (phone login, verification code, forgot password) and decide whether the implementation targets the live backend or a contract-level implementation. Also choose the allowed stub boundary for SMS and rate-limit tests.
  - Requires: A reachable auth API or its OpenAPI spec, a verification-code provider, and a rate-limit fixture for the failure tests.
  - Gate: Wire the three flows, keep English error messages, cover success / failure / rate-limit with unit tests, remove the dead commented code, and pass the app's typecheck and tests.

- [ ] Implement the FRONTEND_PRICEPLES visual and five-dimension quality gates.
  - Problem: Section 6 (`:49-50`) requires an executable visual quality detection mechanism with quantified diff criteria; section 9 (`:68-69`) requires automated coverage of structural correctness, responsive layout, visual consistency, performance metrics and basic accessibility. No visual / screenshot / a11y / performance test file exists anywhere under `apps/`. Existing performance assertions only check `not.toThrow()` (e.g. `AlertFlow.test.tsx:695`, `RealTimeDataDisplay.test.tsx:226`).
  - Suggestion: Add a runnable screenshot-diff / visual regression job with an explicit pixel or perceptual threshold, plus a11y and performance budgets wired into CI; decide the tool per app (Playwright for the two web frontends, a device-matrix snapshot runner for the Expo app).
  - Decision: Choose the tool per app, the numeric visual diff threshold (pixel or perceptual), who approves the baseline screenshots, and the a11y / performance budget numbers. FRONTEND_PRICEPLES sections 6 and 9 demand quantified criteria but specify no values.
  - Requires: Playwright browser binaries and baseline images for the two web frontends, a device-matrix snapshot runner for the Expo app, and a CI job able to store and compare baselines.
  - Gate: Land a runnable screenshot-diff job with an explicit threshold, plus a11y and performance budgets wired into CI, and show one green and one intentionally failing run as evidence.

- [ ] Revoke the exposed classic PAT and purge it from git history.
  - Problem: Commit `3c07b882` (2026-05-26) stores a plaintext `ghp_…` token in `package.json` `repository.url`; the value is still readable from the object database even though the current `package.json`, `.git/config` and `remote.origin.url` are clean and `lefthook.yml:23-28` now rejects new tokens in added lines. The historical token already returns HTTP 401, but it was never revoked or purged. `~/.git-credentials` does not exist on this machine (`credential.helper=manager` stores credentials in the Windows Credential Manager), so the working classic PAT reported on 2026-11-09 must be checked and revoked in GitHub Settings rather than deleted from a file. A CI secret scan was added on 2026-11-12 (`.github/workflows/secret-scan.yml`): it blocks newly introduced secrets on every push and PR and reports full-history findings on manual or scheduled runs.
  - Suggestion: Still open: delete/revoke the classic PATs in GitHub Settings, replace them with a fine-grained PAT or the OS credential manager, and purge the historical value (`git filter-repo` / BFG) with a coordinated force-push and post-purge rotation. If a history rewrite is rejected, record the accepted risk explicitly. Remove this item only after the account-side actions are confirmed.
  - Decision: Authorise (or reject) a git history rewrite. A `filter-repo` / BFG purge changes every commit hash, affects all collaborators, forks, open PRs and CI references, and needs a coordinated force-push. If rejected, the alternative is to record the accepted risk explicitly.
  - Requires: GitHub account access to Settings -> Developer settings to revoke the classic PATs, repository admin rights for a force-push, coordination with every collaborator, and a post-purge credential rotation. The working credential is in the Windows Credential Manager, not `~/.git-credentials`.
  - Gate: Confirm the revocation in GitHub Settings, replace the credential with a fine-grained PAT or the OS credential manager, purge the historical value with a coordinated force-push (or record the accepted risk), and re-run the full-history secret report.

- [ ] Point the performance pipeline at a real staging target.
  - Problem: `GET /repos/lotaway/meta-web-three/actions/variables` returns `total_count: 0`, so `vars.PERF_BASE_URL` is unset and every run still measures the built-in `127.0.0.1:18087` mock; runs #163–#165 are green but never touched a real service. On 2026-11-12 the unconditional `use_mock = True` in the schedule branch (formerly `:127`) was replaced with `use_mock = not base`, mock runs now emit a warning, and the report is labelled MOCK DATA so mock numbers cannot be read as staging results. Those code defects are fixed; the repository variable is still unset and no run has been verified against a real service.
  - Suggestion: Set the `PERF_BASE_URL` repository/environment variable to a reachable staging gateway, re-run a `workflow_dispatch` and one scheduled run against it, and confirm the step output shows the real URL with `use_mock=false` before removing this item.
  - Decision: Authorise load testing against staging, provide the target URL and a test window, and choose whether the daily schedule may hit it or only manual runs.
  - Requires: A staging gateway reachable from `ubuntu-24.04` runners, repository / environment variable write permission for `PERF_BASE_URL`, and `workflow_dispatch` permission.
  - Gate: Set the variable, run one `workflow_dispatch` and one scheduled run, and confirm the step output shows the real URL with `use_mock=false` and no MOCK DATA label.

- [ ] Add reproducible installs and an Electron runtime smoke check for `apps/digital-twin/system-management`.
  - Problem: `d10f4015` added `.github/workflows/system-management-check.yml` (run #1 success), fixed the build context / Dockerfile / compose paths and added `npm run build:native`, `npm run typecheck` and `npm run test:run` in CI, which resolves the earlier "no CI builds it" gap. Still open: the four lockfiles present on the development machine (`apps/backstage-admin/package-lock.json`, `apps/client/yarn.lock`, `apps/client/turbo-module/appsdk/yarn.lock`, `apps/digital-twin/system-management/package-lock.json`) are all gitignored, so a clean checkout has none (the workflow comments at `:34-35` disable the package-manager cache for exactly this reason); `system-management/node_modules` is installed locally but does not make CI reproducible, `docker/Dockerfile:20-24` and the workflow `:49` both use unpinned `npm install --legacy-peer-deps`, neither runs `npm run rebuild:sqlite3` (the app declares `better-sqlite3 ^11.0.0`), and neither launches Electron (`ELECTRON_SKIP_BINARY_DOWNLOAD=1` at `docker/Dockerfile:4` and workflow `:48`). Native ABI regressions and the two insecure `BrowserWindow` configs tracked in "Harden Electron renderer isolation" above are therefore never exercised at runtime. A mirror gap on 2026-11-09 (`npmmirror` missing `@typescript-eslint/parser@8.72.2`) already broke installs; the mirror has since synced, but installs remain unpinned.
  - Suggestion: Commit a lockfile for this app (then for the other `apps/*` packages), switch CI and Docker back to `npm ci`, pin the registry via `.npmrc`, add a CI step for `npm run rebuild:sqlite3` plus a headless Electron launch/smoke (or document the required manual device check), and record the result of a real-device run of both windows before removing this item. Confirm the lockfile decision first: an earlier instruction asked to drop tasks tied to node_modules content or git tracking state.
  - Decision (resolve first): Confirm whether a lockfile may be committed for this app and then the other `apps/*` packages. This conflicts with the earlier instruction to drop tasks tied to node_modules content or git tracking state, so it must be answered explicitly before any install change.
  - Requires: A headless Electron runtime in CI (xvfb, and lifting `ELECTRON_SKIP_BINARY_DOWNLOAD=1`), a pinned registry via `.npmrc`, and a real-device / GUI run of both BrowserWindows for the smoke result.
  - Gate: Commit the lockfile, switch CI and Docker to `npm ci`, add `npm run rebuild:sqlite3`, add a headless Electron launch / smoke step, and record a real run of both windows.

## Completed Tasks (archived)

English notes only. Each item was verified with CHECK_LIST.md before archival.
Per DEVELOPMENT.md step 4, passed tasks are removed from the actionable list and
recorded here as checked items.

- [x] Clear completed and unverifiable tasks from TODO.md. Commit: `57d1db9a`.
- [x] Generate native types before typecheck in CI and Docker; fix the digital-twin build context, Dockerfile, compose paths and add the check workflow. Commit: `d10f4015`.
- [x] Fix the pre-commit guards so they no longer fail on Windows; verify the hooks with staged probes. Commit: `ccb508d1`.
- [x] Make `apps/backstage-admin` env-default tests hermetic with a pure `resolveEnv(vars)` factory and explicit-input tests. The reported 1/5 local failure could not be reproduced (5/5 passed); the real defect was the ambient `.env` dependency. Verified 8/8 under both default and non-default env files, plus `vue-tsc --build`. Commit: `efc88769`.
