# TODO

注意所有修改完成后都要进行对应的项目/服务编译确保没有语法错误、引入库错误等问题。

## 测试补充

### digital-twin-api.test.ts

- [ ] 目标是给 `fetchDevices` 补充 HTTP 调用 Mock 和响应验证（当前仅验证函数存在）
- [ ] 目标是给 `fetchActiveAlerts` 补充 HTTP 调用 Mock 和响应验证
- [ ] 目标是给 `fetchStatsSummary` 补充 HTTP 调用 Mock 和响应验证
- 路径：`apps/digital-twin/system-management/src/renderer/services/digital-twin-api.test.ts`
- 难点：Mock axios 的网络请求，验证请求 URL、请求方法、响应数据结构的完整性

### 端到端测试

- [ ] 目标是实现 3D 场景加载的 E2E 测试，验证场景初始化、模型加载、相机控制功能
- [ ] 目标是实现实时数据展示的 E2E 测试，验证 WebSocket 推送到 UI 更新链路
- [ ] 目标是实现告警流程的 E2E 测试，覆盖告警创建 → 推送 → 前端展示 → 告警确认/关闭全链路
- 难点：E2E 测试框架待确认（Playwright/Cypress），3D 场景测试需处理 WebGL 渲染和动画帧

## 跨服务鉴权体系延伸

### ERP 各服务接入 Gateway

- [ ] 目标是将 ERP 各服务注册到 ZooKeeper，实现 Gateway 自动路由
- [ ] 目标是 ERP 服务消费 Gateway 传递的 `X-User-Id` / `X-User-Role` 请求头
- [ ] 目标是 ERP 服务使用 `@RequirePermission` 注解进行接口鉴权
- [ ] 目标是统一规划 ERP 权限资源树，避免自建鉴权轮子

### 供应链各服务接入 Gateway

- [ ] 目标是将供应链各服务注册到 ZooKeeper，实现 Gateway 自动路由
- [ ] 目标是供应链服务消费 Gateway 传递的 `X-User-Id` / `X-User-Role` 请求头
- [ ] 目标是供应链服务使用 `@RequirePermission` 注解进行接口鉴权
- [ ] 目标是统一规划供应链权限资源树，避免自建鉴权轮子

- [] 按照[MES标准方案参考](TODO_MES_SPEC.md)检查当前项目是否完成了MES必须的一些功能，并以项目为基础进行完善，查漏补缺，优化到接近顶尖产品