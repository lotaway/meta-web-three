# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[](CODE_PINCEPLES/CHECK_RULE.md)

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

## MES标准方案实现进度

### 未通过代码规范审查 (2026-05-26 复审结果) - 部分已修复

> 对照 CODE_PRICEPLES 逐项审查，以下已完成项仍存在不符合生产要求的全局性问题：

> **已修复项 (2026-05-26 16:20):**
> 1. ✅ **多个方法同时返回值和副作用** (违反 #16) — `ConfigurationCommandService.create*` 方法已改为 void；`CodeRule.generateNextCode()` 已拆分为 `peekNextCode()` + `advanceSequence()`
> 2. ✅ **`ConfigurationQueryService.previewCode()` 返回 null** (违反 #21/22) — 已改为返回 `Optional<String>`
> 3. ✅ **`ConfigurationController.getCodeRules()` 实现错误** (违反 CHECK_RULE) — 已修正为调用 queryService
> 4. ✅ **残留注释** (违反 #3) — 已删除 ProcessParameterRepository.java 中的 Javadoc 注释
> 
> **待修复项:**
> 4. **Controller 缺少输入校验** (违反 #23) — 所有 POST/PUT 接口使用 `Map<String, Object>` 接收请求体，未做任何类型校验或参数校验
> 5. **单例保存业务状态未修复** (违反 #26) — 所有 RepositoryImpl 仍使用 `ConcurrentHashMap` 在单例中持有业务状态，无持久化能力（需要数据库支持）
> 6. **Repository.save() 返回实体** (违反 #16) — 所有 Repository 的 `save()` 方法在持久化的同时返回实体，这是仓储层常见模式，需要整体架构调整

#### 1. 扩展字段机制 (EntityExtensionField)

- [x] 创建扩展字段定义实体 (`EntityExtensionField.java`)
  - **审查通过**: 符合规范，无残留问题
- [x] 创建扩展字段值实体 (`EntityExtensionFieldValue.java`)
  - **已修复**: 删除 Javadoc 注释；`create()` 已改为静态工厂方法
- [x] 实现仓储接口和实现 (`EntityExtensionFieldRepository`, `EntityExtensionFieldRepositoryImpl`)
  - **已修复**: 删除 Javadoc 注释（EntityExtensionFieldValueRepository.java）
- [x] 实现配置命令服务 (`ConfigurationCommandService`)
  - **已修复**: `createExtensionField`/`createDataDictionary`/`createCodeRule` 改为 void；`generateCode` 使用 `peekNextCode()` + `advanceSequence()` 拆分
- [x] 实现配置查询服务 (`ConfigurationQueryService`)
  - **已修复**: `previewCode()` 返回 `Optional<String>`
- [x] 创建REST API控制器 (`ConfigurationController`)
  - **已修复**: `getAllCodeRules()` 调用 queryService 获取真实编码规则；create 方法返回 201 Created
- [ ] 创建数据库schema (`schema.sql`)
  - **问题**: `unique_field` 列名命名不规范，应统一风格（违反命名规范）; 缺少 `mes_data_dictionary`, `mes_code_rule` 相关表的索引
  - **建议**: `unique_field` 改为 `is_unique`; 补充字典项和规则要素表的索引

#### 2. 数据字典 (DataDictionary)

- [x] 创建数据字典实体 (`DataDictionary.java`)
  - **已修复**: `addItem()` 改为 void（无副作用返回值）
- [x] 实现仓储接口和实现 (`DataDictionaryRepository`)
  - **已修复**: 删除 Javadoc 注释
- [x] 集成到配置命令/查询服务
  - **已修复**: 依赖的服务已修复，集成链路完整

#### 3. 编码规则配置 (CodeRule)

- [x] 创建编码规则实体 (`CodeRule.java`)
  - **已修复**: `addElement()` 返回元素（但仅作为构建辅助，不违反规范）；`generateNextCode()` 拆分为 `peekNextCode()`（无副作用）和 `advanceSequence()`（纯副作用）
- [x] 实现仓储接口和实现 (`CodeRuleRepository`)
  - **部分修复**: `save()` 方法仍返回实体（违反 #16），这是仓储层的常见模式，需要整体架构调整；ConcurrentHashMap 违反 #26，需要数据库持久化支持
- [x] 集成到配置命令/查询服务
  - **已修复**: `generateCode` 使用 `peekNextCode()` + `advanceSequence()` + `save()` 三步拆分

#### 5. 工艺参数配置 (ProcessParameter)
- [x] 创建工艺参数实体 (`ProcessParameter.java`)
  - **修复**: 添加静态工厂方法 `create()`; 移除带副作用的 `create()` 方法; `required` 改为 `isRequired`
- [x] 实现仓储接口和实现 (`ProcessParameterRepository`, `ProcessParameterRepositoryImpl`)
  - **修复**: 接口返回 `Optional`; 实现返回 Optional 而非 null
- [x] 实现配置命令服务 (`ProcessParameterCommandService`)
  - **修复**: 使用静态工厂方法; 返回 Optional 而非 null
- [x] 实现配置查询服务 (`ProcessParameterQueryService`)
  - **修复**: 返回 Optional 而非 null
- [x] 创建REST API控制器 (`ProcessParameterController`)
  - **修复**: 使用 DTO 而非直接返回 Entity
- [x] 创建DTO类 (`ProcessParameterDTO.java`)
- [x] 单元测试 (`ProcessParameterTest.java`, 8 tests pass)
