# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

本轮已完成的 8 项任务与 5 项配套改进均通过 CHECK_LIST.md 检查，按规则 1 已从 TODO 中移除。
完成内容与验证记录见 git 提交 `50da349e` 与本轮提交。

「遗留问题」4 项已于 2026-11-09 全部通过 CHECK_LIST.md 检查并移除，验证记录：
- `apps/client`：`npx jest --runInBand` → 8 suites / 99 tests 全绿，ESLint 对全部改动文件 0 error / 0 warning；显式声明了此前未写入 `package.json` 的 `@testing-library/react-native`。
- `apps/tools/DownloadGraphQLSchema.js`：用 mock introspection 端到端生成了 `type Query { hello: String }` SDL；`generate-api.js` 在 schema 下载失败时仍以 0 退出并跳过 codegen。
- `apps/backstage-admin/src/utils/http.ts`：`@typescript-eslint/ban-ts-comment` + `no-explicit-any` 从 6 errors 降为 0，隔离 tsc 检查本文件无报错。
- `getApiBaseUrl()` 单测：8 个用例覆盖环境变量优先级 / 空值 / hostUri / Android / iOS 兜底。

「CI 流水线收尾任务」中已通过检查并移除：
- **安装并启用 lefthook**：`lefthook.yml` 由全注释示例改为 3 个真实 pre-commit 守卫（空白/冲突标记、`.workbuddy/` 入库、明文 GitHub token），根 `package.json` 增加 `lefthook` devDependency 与 `prepare: lefthook install`；本地 `lefthook install` 生成 hook，实测能拦截带尾随空格与假 token 的暂存文件。
- **确认每日定时 smoke 转绿**：`Performance Testing Pipeline` 定时运行 `#160` / `#161` / `#162` 均为 success。

### CI 流水线收尾任务（尚待验证或需外部操作）

本轮 CI 修复与线上验证记录见提交 `5b2bc0fd` / `20f9417e` / `8566a1f5` / `9696c501`，以及 `Performance Testing Pipeline #159`、`Core Services Tests #11`–`#13`。

- [ ] **确认升级后的 workflows 回归通过**：三个 workflow 的 Actions 已升级到当前主版本（`checkout@v7` / `setup-java@v6` / `cache@v6` / `upload-artifact@v7` / `setup-python@v7`），`runs-on` 固定为 `ubuntu-24.04`；`domain-boundary-check.yml` 已补 `master` 分支过滤（该 workflow 历史运行数为 0）。**问题定位**：这些改动只有推送到 `master` 后才会被 GitHub 执行。**建议修改**：推送后确认 `Domain Boundary Check` 首次运行为 success，`Core Services Tests` 三个 job 全绿，并手动 `workflow_dispatch` 一次 `Performance Testing Pipeline`；全部通过后再按 CHECK_LIST.md 移除本项。
- [ ] **清理失效的明文 GitHub PAT 并统一凭证管理**：`remote.origin.url` 内嵌的失效 `ghp_bk…Bqkn` 已从本地移除，统一改用 credential helper；新加的 lefthook 守卫会阻止明文 token 再次进入提交。**问题定位**：旧 token 的 revoke 与 credential helper / SSH 的最终选型需要在 GitHub 账号侧操作，本地无法完成。**建议修改**：在 GitHub → Settings → Developer settings 中 revoke 旧 classic PAT，确认新 token 同时具备 `repo` + `workflow` scope，然后复测 `git push` 与 REST 鉴权。
- [ ] **把性能流水线接到真实环境**：仓库 Actions Variables 为空，`vars.PERF_BASE_URL` 未配置，`Performance Testing Pipeline` 目前只打 workflow 内置的 Python mock API。**问题定位**：#159–#162 全绿只证明流程可用，阈值与响应解析尚未经过真实接口检验。**建议修改**：在仓库 Settings → Actions → Variables 配置 staging 的 `PERF_BASE_URL`（如 `https://staging.example.com:8443`），或用 service container / 自托管 runner 起真实服务，再手动 `workflow_dispatch` 验证阈值脚本对真实响应的解析。
- [ ] **确认 Dependabot 告警在大版本升级后关闭**：10 个告警（1 high、5 moderate、4 low）已按官方修复版本升级：`electron 41.10.7 → 42.11.16`（一次覆盖 8 个 Electron 告警）、`vue-i18n ^9.14.0 → ^11.4.14`、`vitest ^3.2.7 → ^4.1.11`（另把 system-management 的 `^4.1.7` 提到 `^4.1.11`）。升级前的 API/配置审计结论与适配改动见本轮提交：**Electron 42** 未使用 `clearStorageData.quotas` / `nativeImage.createFromNamedImage` / `offscreen` / `ELECTRON_SKIP_BINARY_DOWNLOAD`，但 macOS 通知改为 `UNNotification` 且要求签名，已在 `notification.ts` 增加 `onerror` 兜底并接上原本未使用的 `onClick`；**vue-i18n 11** 要求真实安装的 Composition API 实例，已在 `src/locales/index.ts` 补 `createI18n({ legacy: false, globalInjection: true, escapeParameter: true })` 并在 `main.ts` 注册（此前 87 处 `useI18n()` 没有实例）；**Vitest 4** 仅用到 `environment` / `include`，无 `coverage.all` / `poolOptions` / `deps.*` 等被移除的配置。**问题定位**：这两个应用都没有 lockfile、本地也未安装依赖，且现有三条 CI workflow 都不构建它们。**建议修改**：推送后等 Dependabot 重新扫描确认告警关闭；在具备依赖环境时分别跑 `vue-tsc --build` + `vitest run`（backstage-admin）与 `npm run typecheck` + `electron-rebuild -f -w better-sqlite3`（system-management）。注意 `@electron/remote` 的 peer 为 `electron >= 13.0.0`，Electron 42 下 `nodeIntegration: true` + `contextIsolation: false` 的两处 `BrowserWindow` 配置仍需真机回归。
