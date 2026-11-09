# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

本轮已完成的 8 项任务与 5 项配套改进均通过 CHECK_LIST.md 检查，按规则 1 已从 TODO 中移除。
完成内容与验证记录见 git 提交 `50da349e` 与本轮提交。

### 本轮已通过 CHECK_LIST.md 检查并移除的任务

- 「遗留问题」4 项：`apps/client` `npx jest --runInBand` 8 suites / 99 tests 全绿、改动文件 ESLint 0 error / 0 warning，并显式声明了此前漏写的 `@testing-library/react-native`；`DownloadGraphQLSchema.js` 用 mock introspection 端到端生成 SDL，`generate-api.js` 在 schema 下载失败时仍以 0 退出；`http.ts` 的 6 处 ESLint 违规降为 0 并补齐 4 个 API 的响应泛型；`getApiBaseUrl()` 新增 8 个单测。
- **安装并启用 lefthook**：`lefthook.yml` 由全注释示例改为 3 个真实 pre-commit 守卫（空白/冲突标记、`.workbuddy/` 入库、明文 GitHub token），根 `package.json` 增加 `lefthook` devDependency 与 `prepare: lefthook install`；实测能拦截带尾随空格与假 token 的暂存文件，提交时 3 个守卫全部执行通过。
- **确认每日定时 smoke 转绿**：`Performance Testing Pipeline` 定时运行 `#160` / `#161` / `#162` 均为 success。
- **升级 Actions 并回归三个 workflow**：`checkout@v7` / `setup-java@v6` / `cache@v6` / `upload-artifact@v7` / `setup-python@v7`，`runs-on` 固定 `ubuntu-24.04`；`Domain Boundary Check #1`（历史首次运行）10 步全绿、`Core Services Tests #14` 三个 job 全绿、`Performance Testing Pipeline #163` 16 步全绿。
- **Dependabot 告警**：10 个告警（1 high、5 moderate、4 low）在 `e341220a` 推送后重新扫描，当前 open 数为 **0**。升级前完成了 API/配置审计并先落地适配代码：`electron 41.10.7 → 42.11.16`（一次覆盖 8 个 Electron 告警，并为 macOS `UNNotification` 增加 `onerror` 兜底）、`vue-i18n ^9.14.0 → ^11.4.14`（补 `createI18n({ legacy: false, globalInjection: true, escapeParameter: true })` 并在 `main.ts` 注册，此前 87 处 `useI18n()` 没有实例）、`vitest ^3.2.7 → ^4.1.11`（仅用 `environment` / `include`，无被移除的配置项，另把 system-management 的 `^4.1.7` 提到 `^4.1.11`）。`apps/backstage-admin` 已用真实依赖跑通 `vue-tsc --noEmit`（0 error）、ESLint（0 error）与 Vitest 4（5/5 用例通过）。

### 仍需处理（需外部操作或额外环境）

- [ ] **清理失效的明文 GitHub PAT 并统一凭证管理**：`remote.origin.url` 内嵌的失效 `ghp_bk…Bqkn` 已从本地移除，统一改用 credential helper；新加的 lefthook 守卫会阻止明文 token 再次进入提交。**问题定位**：旧 token 的 revoke 与 credential helper / SSH 的最终选型需要在 GitHub 账号侧操作，本地无法完成。**建议修改**：在 GitHub → Settings → Developer settings 中 revoke 旧 classic PAT，确认新 token 同时具备 `repo` + `workflow` scope，然后复测 `git push` 与 REST 鉴权。
- [ ] **把性能流水线接到真实环境**：仓库 Actions Variables 为空，`vars.PERF_BASE_URL` 未配置，`Performance Testing Pipeline` 目前只打 workflow 内置的 Python mock API。**问题定位**：#159–#163 全绿只证明流程可用，阈值与响应解析尚未经过真实接口检验。**建议修改**：在仓库 Settings → Actions → Variables 配置 staging 的 `PERF_BASE_URL`（如 `https://staging.example.com:8443`），或用 service container / 自托管 runner 起真实服务，再手动 `workflow_dispatch` 验证阈值脚本对真实响应的解析。
- [ ] **在具备依赖环境时回归 `apps/digital-twin/system-management`**：该应用没有 lockfile、本地未安装依赖，且现有三条 CI workflow 都不构建它，Electron 42 的兼容性目前只做了静态审计（未使用 `clearStorageData.quotas` / `nativeImage.createFromNamedImage` / `offscreen` / `ELECTRON_SKIP_BINARY_DOWNLOAD`，`@electron/remote` peer 为 `electron >= 13.0.0`）。**建议修改**：执行 `npm install`、`npm run typecheck`、`electron-rebuild -f -w better-sqlite3`，并真机验证两处 `nodeIntegration: true` + `contextIsolation: false` 的 `BrowserWindow` 配置、macOS 通知签名行为，以及 `Dockerfile` 中依赖 `package-lock.json` 的 `npm ci` 步骤。
