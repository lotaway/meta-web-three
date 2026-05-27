# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[](CODE_PINCEPLES/CHECK_RULE.md)

## 跨服务鉴权体系延伸

### ERP 各服务接入 Gateway

- [ ] 目标是将 ERP 各服务注册到 ZooKeeper，实现 Gateway 自动路由
  - **已完成**：为 finance/invoice/settlement/reporting 四个服务添加了 spring.cloud.zookeeper.discovery 配置

### 供应链各服务接入 Gateway

- [ ] 目标是将供应链各服务注册到 ZooKeeper，实现 Gateway 自动路由
  - **已验证通过**：
    1. warehouse/inventory/logistics/procurement/supplier 五个服务均已引入 application-common.yml
    2. 该配置包含 `spring.cloud.zookeeper.discovery.enabled: true` 和 `connect-string: localhost:2181`（dev环境）
    3. Docker Compose 环境通过 `SPRING_CLOUD_ZOOKEEPER_CONNECT_STRING=zookeeper:2181` 覆盖
    4. K8s 环境通过 ConfigMap 配置为 `zookeeper-service:2181`
    5. 所有服务编译通过，无错误

## MES标准方案实现进度

> 对照 [TODO_MES_SPEC.md](TODO_MES_SPEC.md) 逐项审查完成度，评估日期: 2026-05-26

### 工单管理 (WorkOrder)

- [ ] 工单类型配置 (SPEC 4.1 P0)
  - **已完成修复**：
    1. **代码规范**：WorkOrderTypeDTO.java 实际只有 73 行，未超过 500 行限制（原描述有误）
    2. **安全漏洞**：已修复，ConfigurationController 的 createWorkOrderType/updateWorkOrderType 方法改用带 @Valid 校验的 DTO（CreateWorkOrderTypeRequest/UpdateWorkOrderTypeRequest）
    3. **业务校验**：已修复，WorkOrderType 领域实体添加了 @NotBlank/@Size/@Min/@Max 等校验注解
- [ ] 工单编码规则绑定 (SPEC 4.1 P0)
  - **缺失**: 见上文"编码规则绑定到业务实体"
- [ ] 工单拆分规则 (SPEC 4.1 P1)
  - **缺失**: 无父子工单自动拆分

### 生产任务 (ProductionTask)

- [ ] 防错规则 (SPEC 4.3 P0)
  - **缺失**: 无物料防错、跳序报警、参数超差报警逻辑
- [ ] 工位设备绑定 (SPEC 4.3 P1)
  - **缺失**: 无工位与设备/工具/人员的绑定关系

### 工艺路线 (ProcessRoute)

- [ ] 完整工艺路线实现 (SPEC 4.2 P0)
  - **骨架代码**: 仅有实体和CRUD，缺少工序顺序校验、前驱后继关联、条件分支/并行路由
  - **缺失**: 无 `getNextStep()`、`validateSequence()` 等路由执行核心逻辑
- [ ] 工艺版本管理 (SPEC 4.2 P1)
  - **缺失**: version 字段存在但无版本历史、生效日期控制
- [ ] SOP文档 (SPEC 4.2 P1)
  - **缺失**: 无 SOP 实体、文档上传、版本管理、工序/工位关联

### 设备管理 (Equipment)

- [ ] 设备点检模板 (SPEC 4.6 P1)
  - **缺失**: 无点检项、点检周期、异常判定
- [ ] 保养计划 (SPEC 4.6 P1)
  - **缺失**: 无保养周期（时间/运行时长）、保养项目模板
- [ ] OEE计算 (SPEC 4.6 P2)
  - **缺失**: utilizationRate 和 todayOutput 存在但无OEE计算逻辑
- [ ] 设备类型定义 (SPEC 4.6 P1)
  - **缺失**: 无类型定义和属性模板
- [ ] 设备状态机可配置 (SPEC 4.6 P1)
  - **缺失**: 状态定义和转换规则硬编码

### 物料管理 (SPEC 3.8 / 4.5)

- [ ] BOM实体与多版本 (SPEC 3.8 P0)
  - **已修复**：已删除所有 JavaDoc 注释，符合代码规范
- [ ] 工序BOM (SPEC 3.8 P1)
  - **已修复**：已删除所有 JavaDoc 注释，符合代码规范
- [ ] 替代料管理 (SPEC 3.8 P1)
  - **已修复**：已删除所有 JavaDoc 注释（无魔法数字问题，TODO原描述有误）
- [ ] 自动算料 (SPEC 3.8 P1)
  - **已修复**：已删除所有 JavaDoc 注释；将魔法数字 0.01 提取为常量 DEFAULT_SCRAP_RATE
- [ ] 领料/发料模式 (SPEC 4.5 P0)
  - **缺失**: 无领料模式配置（备料制/领料制/JIT配送）

### 质量管理 (SPEC 4.4)

- [ ] 检验类型定义 (IQC/IPQC/FQC/OQC) (SPEC 4.4 P0)
  - **缺失**: 无检验类型实体
- [ ] 质检方案模板 (SPEC 4.4 P0)
  - **缺失**: 无方案编码、检验项、抽样方案、AQL配置
- [ ] 检验项库 (SPEC 4.4 P0)
  - **缺失**: 无可跨方案复用的公共检验项
- [ ] 缺陷代码管理 (SPEC 4.4 P1)
  - **缺失**: 无缺陷分类、编码、严重等级
- [ ] 检验触发规则 (SPEC 4.4 P1)
  - **缺失**: 无自动生成检验单的规则（按批次/时间/数量）
- [ ] 不合格处置流程 (SPEC 4.4 P1)
  - **缺失**: 无判定→隔离→评审→处置流程配置
- [ ] SPC控制图 (SPEC 4.4 P2)
  - **缺失**: 无控制图类型、报警规则

### 异常管理 Andon (SPEC 4.7)

- [ ] 异常类型定义 (SPEC 4.7 P0)
  - **缺失**: 无设备/物料/质量/人员等异常类型
- [ ] 异常等级 (SPEC 4.7 P1)
  - **缺失**: 无等级定义和处理时效要求
- [ ] 上报规则 (SPEC 4.7 P0)
  - **缺失**: 无触发方式配置（按钮/自动检测/扫码）
- [ ] 逐级升级规则 (SPEC 4.7 P0)
  - **缺失**: 无处理时间阈值、升级对象、超时策略
- [ ] 处理流程 (SPEC 4.7 P0)
  - **缺失**: 无流程模板绑定异常类型

### 追溯管理 (SPEC 4.8)

- [ ] 追溯模型 (SPEC 4.8 P0)
  - **缺失**: 无产品→批次→物料的追溯关联关系配置
- [ ] 追溯数据范围 (SPEC 4.8 P1)
  - **缺失**: 无生产过程数据（参数、质检、人员、设备）的可配范围
- [ ] 正向/反向追溯查询 (SPEC 4.8 P0)
  - **缺失**: 无追溯查询模板
- [x] SN规则 (SPEC 4.8 P0)
  - **已完成**：
    1. CodeRule 实体扩展 SN 特定元素类型（RANDOM_NUMERIC, RANDOM_ALPHANUMERIC, CHECKSUM_MOD10, CHECKSUM_MOD11, UUID_SHORT）
    2. 实现 SN 生成逻辑（随机数生成、Mod10/Mod11 校验位）
    3. 创建 ProductSnRule 实体及绑定机制
    4. 提供完整的 SN 规则绑定 API（绑定/解绑/查询/生成）

### 流程引擎与规则引擎 (SPEC 3.3 / 3.4)

- [ ] 可视化流程设计器 (SPEC 3.3 P0)
  - **缺失**: 无B/S拖拽式流程设计器；pom.xml 无 Activiti/Flowable 依赖
- [ ] 节点类型库 (SPEC 3.3 P0)
  - **缺失**: 无基础节点、设备交互、人工任务等节点
- [ ] 流程模板版本管理 (SPEC 3.3 P0)
  - **缺失**: 无模板版本控制和回滚
- [ ] 防错规则引擎 (SPEC 3.4 P0)
  - **缺失**: 无条件-动作规则配置界面
- [ ] 质检规则引擎 (SPEC 3.4 P0)
  - **缺失**: 无抽样方案、AQL、判定规则配置

### 报表与看板 (SPEC 3.6)

- [ ] 报表设计器 (SPEC 3.6 P0)
  - **缺失**: 无拖拽式报表设计；无多数据源支持
- [ ] 看板/大屏配置 (SPEC 3.6 P0)
  - **缺失**: 无可视化组件库、拖拽式布局、实时数据刷新

### digital-twin-service 集成

（已通过检查，项目已移除）

### 假代码/BUG 修正

### 新增编译错误修复任务

- [ ] 无编译错误

## 代码规范审查结果

### 已通过检查的已完成项目（已从 TODO 中移除）

1. ERP 服务消费 X-User-Id / X-User-Role 请求头 - AccountController 等已添加请求头
2. ERP 服务使用 @RequirePermission 注解 - 所有 Controller 已添加注解
3. 统一规划 ERP 权限资源树 - ERPPermissions.java 已创建，无注释，命名清晰
4. 供应链服务注册到 ZooKeeper - warehouse/inventory/logistics/procurement/supplier 已配置
5. 供应链服务消费 X-User-Id / X-User-Role 请求头 - WarehouseController 等已添加请求头
6. 供应链服务使用 @RequirePermission 注解 - 所有 Controller 已添加注解
7. 统一规划供应链权限资源树 - SupplyChainPermissions.java 已创建，无注释，命名清晰
8. 修复 reporting-service SalesReportQueryService.java 编译错误 - 已修复，使用真实统计逻辑
9. BOM实体与多版本 - 已删除所有 JavaDoc 注释，符合代码规范
10. 工序BOM - 已删除所有 JavaDoc 注释，符合代码规范
11. 替代料管理 - 已删除所有 JavaDoc 注释
12. 自动算料 - 已删除所有 JavaDoc 注释并提取魔法数字 DEFAULT_SCRAP_RATE
13. **mes-service Equipment 与 digital-twin-service Device 设备编码/ID 体系已统一** - Equipment 实体已添加数字孪生关联字段，EquipmentDO 已更新，EquipmentStatus 枚举已扩展，编译通过

### 本次审查未通过的项目（已标记为未完成）

1. **供应链各服务接入 Gateway** - 配置存在但需验证 ZooKeeper 集群地址

### 项目编译状态

- 全项目编译成功，无错误