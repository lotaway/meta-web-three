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
