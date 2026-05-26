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

### 未通过代码规范审查 (2026-05-26 审查结果) - 已修复部分

> 对照 CODE_PRICEPLES 审查，存在以下全局性问题导致已完成项不符合生产要求：
> 1. **缺少单元测试** (违反 #33) — 整个模块无任何测试
> 2. **接口层泄露实现细节** (违反 #37) — Controller 直接返回 Entity，应使用 DTO
> 3. **单例保存业务状态** (违反 #26) — Repository 用 `ConcurrentHashMap` 在单例中持有业务状态
> 4. **返回 null 表达错误** (违反 #21/22) — 多处返回 null 代替 `Optional` 或异常
> 5. **create 方法同时返回值和副作用** (违反 #16) — `createExtensionField/createDataDictionary/createCodeRule` 等均违反
>
> **修复完成情况 (2026-05-26 15:41):**
> - ✅ 创建 DTO 类: EntityExtensionFieldDTO, DataDictionaryDTO, CodeRuleDTO, EntityExtensionFieldValueDTO
> - ✅ 实体类重构: 添加静态工厂方法 (create), 移除冗余注释, unique → isUnique
> - ✅ Repository 接口修改: findById/findByEntityTypeAndFieldCode 返回 Optional
> - ✅ 修复 null 返回: ConfigurationQueryService 返回 Optional 而非 null
> - ✅ 修复 create 方法: 使用静态工厂方法替代带副作用的 create()
> - ✅ 单元测试: 添加 EntityExtensionFieldTest, DataDictionaryTest, CodeRuleTest (14 tests pass)
> - ⚠️ 单例保存业务状态: Repository 仍使用 ConcurrentHashMap (需改用 MyBatis-Plus + PostgreSQL)

#### 1. 扩展字段机制 (EntityExtensionField)
- [x] 创建扩展字段定义实体 (`EntityExtensionField.java`)
  - **修复**: 删除冗余注释; 添加静态工厂方法 `create()`; `unique` 改为 `isUnique`
- [ ] 创建扩展字段值实体 (`EntityExtensionFieldValue.java`)
  - **问题**: 同上
- [x] 实现仓储接口和实现 (`EntityExtensionFieldRepository`, `EntityExtensionFieldRepositoryImpl`)
  - **修复**: 接口返回 `Optional`; 实现保持内存存储(待改数据库)
- [x] 实现配置命令服务 (`ConfigurationCommandService`)
  - **修复**: 使用静态工厂方法; 拆分 create 调用
- [x] 实现配置查询服务 (`ConfigurationQueryService`)
  - **修复**: 返回 `Optional` 而非 null
- [x] 创建REST API控制器 (`ConfigurationController`)
  - **修复**: 使用 DTO; 补全 `GET /code-rules` 调用 queryService
- [ ] 创建数据库schema (`schema.sql`)
  - **问题**: `unique_field` 列名命名不规范，应统一风格（违反命名规范）; 缺少 `mes_data_dictionary`, `mes_code_rule` 相关表的索引
  - **建议**: `unique_field` 改为 `is_unique`; 补充字典项和规则要素表的索引

#### 2. 数据字典 (DataDictionary)
- [x] 创建数据字典实体 (`DataDictionary.java`)
  - **修复**: 添加静态工厂方法; 删除冗余注释
- [x] 实现仓储接口和实现 (`DataDictionaryRepository`)
  - **修复**: 接口已返回 Optional
- [x] 集成到配置命令/查询服务
  - **修复**: 使用静态工厂方法

#### 3. 编码规则配置 (CodeRule)
- [x] 创建编码规则实体 (`CodeRule.java`)
  - **修复**: 添加静态工厂方法; 使用 DateTimeFormatter; 删除冗余注释
- [x] 实现仓储接口和实现 (`CodeRuleRepository`)
  - **修复**: 接口已返回 Optional
- [x] 集成到配置命令/查询服务
  - **修复**: 使用静态工厂方法

#### 4. 项目配置
- [x] 更新mes-service pom.xml添加mybatis-plus依赖
  - **审查通过**: 依赖已存在
- [x] 编译验证通过 (BUILD SUCCESS)
  - **修复**: 编译成功，单元测试通过 (14 tests pass)

#### 5. 工艺参数配置 (ProcessParameter)
- [ ] 创建工艺参数实体 (`ProcessParameter.java`)
- [ ] 实现仓储接口和实现 (`ProcessParameterRepository`, `ProcessParameterRepositoryImpl`)
- [ ] 实现配置命令服务 (`ProcessParameterCommandService`)
- [ ] 实现配置查询服务 (`ProcessParameterQueryService`)
- [ ] 创建REST API控制器 (`ProcessParameterController`)