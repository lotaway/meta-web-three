# TODO

## 永久性规范（持续约束，不作为一次性任务移除）
- [后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES) - 服务端代码规范
- [前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES) - 前端代码规范
- [区块链合约规范](CODE_PINCEPLES/BLOCK_CHAIN_CONTRACT_PRICEPLES) - 区块链合约规范

## 可执行任务（完成后按 DEVELOPMENT.md 流程移除）

本轮已完成的 8 项任务与 5 项配套改进均通过 CHECK_LIST.md 检查，按规则 1 已从 TODO 中移除。
完成内容与验证记录见 git 提交 `50da349e` 与本轮提交。

### 遗留问题（检查中发现，尚未处理）

- [ ] **apps/client 的 jest 测试套件无法启动**：`npx jest` 报 `TypeError: Object.defineProperty called on non-object`，7 个 suite 全部启动失败（0 个用例执行）。该问题在改动前即存在（stash 验证一致），属既有 jest 环境问题。**问题定位**：`apps/client/jest.setup.js` / `jest.setupAfterEnv.js` 在 node 环境中对不可扩展对象调用 `Object.defineProperty`，常见于 mock 目标为 `import.meta` 或模块命名空间对象。**建议修改**：将 `Object.defineProperty` 调用目标改为可扩展的普通对象或使用 `Object.assign`；为 setup 文件补 `test/` 目录下的最小用例回归，确认 7 个 suite 可运行。
- [ ] **apps/tools/DownloadGraphQLSchema.js 缺少 graphql 模块依赖**：执行时报 `Cannot find module 'graphql'`，GraphQL schema 下载失败，`yarn generate:api` 因此以非零码退出。**问题定位**：`apps/tools/` 目录没有独立的 `package.json`，Node 解析不到 `apps/client/node_modules` 中的 `graphql`。**建议修改**：在 `apps/tools/` 增加 `package.json` 声明 `graphql` 依赖，或在 `DownloadGraphQLSchema.js` 中用 `createRequire` 显式解析 client 的 `node_modules` 路径；同时把 `generate-api.js` 的 GraphQL 下载失败改为不阻断 REST 客户端生成。
- [ ] **apps/backstage-admin/src/utils/http.ts 存在 ESLint 违规**：`@typescript-eslint/ban-ts-comment` 报 1 处（`@ts-ignore` 应改为 `@ts-expect-error`），`@typescript-eslint/no-explicit-any` 报 5 处（`http.get/post/put/delete/patch` 的 `<T = any>` 泛型默认）。**问题定位**：既有代码为规避 axios 拦截器返回类型推断问题使用了 `any` 与 `@ts-ignore`。**建议修改**：为 `http` 便捷方法定义 `T = unknown` 默认泛型并用 `CommonResult<T>` 约束返回值，将 `@ts-ignore` 换成 `@ts-expect-error` 并补充具体错误原因说明。
- [ ] **apps/client 缺少可用的单元测试框架**：client 已有 `jest.config.js` 与 `jest.setup.js`，但 7 个 suite 全部启动失败（见上条），当前无任何可执行用例覆盖 `src/lib/api/baseUrl.ts` 的网关地址推导逻辑。**建议修改**：修复 setup 后为 `getApiBaseUrl()` 补单元测试，覆盖 EXPO/NEXT 环境变量优先级、空字符串视为未配置、hostUri 推导与 Android/iOS 兜底分支。

### CI 流水线收尾任务（本轮修复后遗留，尚未处理）

本轮 CI 修复与线上验证记录见提交 `5b2bc0fd` / `20f9417e` / `8566a1f5`，以及 `Performance Testing Pipeline #159`、`Core Services Tests #11` / `#12`。以下均为未处理项，完成后按 DEVELOPMENT.md 规则 4 经 CHECK_LIST.md 验证再移除。

- [ ] **清理失效的明文 GitHub PAT 并统一凭证管理**：本轮 `git push` 曾因 classic PAT 缺 `workflow` scope 被拒；补 scope 后已成功。**问题定位**：`remote.origin.url` 内嵌的 `ghp_bk…Bqkn` 已失效（API 401 Bad credentials），实际生效的是 `~/.git-credentials` / macOS 钥匙串里的 token；GitHub 对 git push 与 REST contents API 修改 `.github/workflows/` 同样要求 `workflow` scope，无法靠换接口绕过。本仓库已固化 `http.version=HTTP/1.1` 以规避本地代理对 github.com 的 HTTP/2 帧错误。**建议修改**：从 remote URL 移除明文 token 并 revoke 旧 token，统一改用 credential helper 或 SSH；清理后复测一次 `git push` 与 API 鉴权。
- [ ] **`domain-boundary-check.yml` 从未在默认分支运行**：仓库默认分支是 `master`，但该 workflow 只在 `push: [main, develop, 'feature/**', 'fix/**']` 与 `pull_request: [main, develop]` 触发。**问题定位**：分支过滤漏掉 `master`，导致领域边界检查历次提交均未执行。**建议修改**：把 `master` 加入 push / pull_request 分支列表（或改为跟随默认分支），推送后确认该 workflow 首次成功执行。
- [ ] **升级 Actions 依赖、消除弃用告警**：run #159 的 annotations 显示 `actions/cache@v4`、`actions/checkout@v4`、`actions/setup-java@v4`、`actions/upload-artifact@v4` 仍面向 Node.js 20 被强制运行在 Node.js 24；`setup-java@v4` 已停止维护建议迁至 `v5`；`ubuntu-latest` 自 2026-10-19 起逐步迁移到 Ubuntu 26。**问题定位**：这些目前只是 warning、不影响 CI 结论，但 runner 镜像与 action 主版本漂移后可能导致构建行为变化。**建议修改**：升级到各 action 当前主版本，将 `runs-on` 固定到已验证的 `ubuntu-24.04`，并回归三个 workflow。
- [ ] **安装并启用 lefthook，恢复提交前检查**：仓库根目录已有 `lefthook.yml`，但本机没有 `lefthook`，本轮每次 commit 都输出 `Can't find lefthook in PATH`，`core.hooksPath` 也未设置。**问题定位**：提交钩子被静默跳过，本地质量门禁形同虚设。**建议修改**：安装 lefthook 并执行 `lefthook install`（或把同等校验搬进 CI），确认提交时钩子真实执行且通过。
- [ ] **把性能流水线接到真实环境**：仓库 Actions Variables 为空，`vars.PERF_BASE_URL` 未配置，当前 `Performance Testing Pipeline` 只打 workflow 内置的 Python mock API，压测数据只验证流水线自身、不反映真实服务性能。**问题定位**：#159 全绿证明流程可用，但阈值与响应解析尚未经过真实接口检验。**建议修改**：为 staging 配置 `PERF_BASE_URL`（如 `https://staging.example.com:8443`），或用 service container / 自托管 runner 起真实服务，再手动 `workflow_dispatch` 验证阈值脚本对真实响应的解析。
- [ ] **确认每日定时 smoke 在新 workflow 下转绿**：手动触发的 #159 已 12 步全绿，但 `cron: '0 2 * * *'`（10:00 GMT+8）的定时运行在修复后尚未执行，上一次定时 #157 仍是旧代码失败。**建议修改**：下一次定时运行后确认 `Performance Testing Pipeline` 为 success，并核对 `schedule` 分支走的是 5 threads × 100 loops 的 mock smoke；确认无误后按 CHECK_LIST.md 规则移除本项。
- [ ] **处理 Dependabot 依赖告警**：`git push` 时 GitHub 提示默认分支存在 10 个漏洞（1 high、5 moderate、4 low）。**问题定位**：本轮只修 CI 编排，未触碰依赖版本。**建议修改**：在 Security → Dependabot 逐项评估，优先处理 high，升级后跑一遍 `Core Services Tests` 与性能流水线回归。
