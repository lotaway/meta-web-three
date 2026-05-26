# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[](CODE_PINCEPLES/CHECK_RULE.md)

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

## MES标准方案实现进度

> 对照 [TODO_MES_SPEC.md](TODO_MES_SPEC.md) 逐项审查完成度，评估日期: 2026-05-26

### 全局架构问题（影响所有模块）
1. **所有 Repository 使用内存存储** — [ ] **[未通过审查]** 已将以下Repository迁移到MyBatis-Plus持久化：ProductionTask、Equipment、EntityExtensionField、EntityExtensionFieldValue、ProcessRoute（原本已持久化）
   - **问题**: 代码中存在大量注释，违反"禁止注释"规范（如 `// ========== DO 与 Entity 转换方法 ==========`、`// 使用 stepCode 映射到 processCode` 等）
   - **建议修改**: 删除所有注释代码，包括分隔注释和字段映射注释
2. **MES 事件系统是假实现** — [ ] **[未通过审查]** `MesEventPublisher` 已接入 Spring ApplicationEventPublisher，事件可正常发布
   - **问题**: 事件使用 HashMap 而不是强类型事件类，不符合"事件、订阅、状态机等机制必须显式建模"的规范；事件类型使用字符串常量而不是枚举，违反"事件名必须使用枚举或唯一常量"的规范
   - **建议修改**: 创建具体的事件类（如 WorkOrderCreatedEvent），使用枚举定义事件类型

### 工单管理 (WorkOrder)

- [ ] 工单状态机需要可配置 (SPEC 4.1 P0)
  - **缺失**: 状态定义为硬编码枚举，无后台配置界面
- [ ] 工单类型配置 (SPEC 4.1 P0)
  - **缺失**: 无工单类型定义（正常/返工/维修），各类型关联不同流程模板
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
  - **缺失**: 整个 factory-domain 无 BOM 或物料相关代码
- [ ] 工序BOM (SPEC 3.8 P1)
  - **缺失**: 无法按工序定义物料清单
- [ ] 替代料管理 (SPEC 3.8 P1)
  - **缺失**: 无物料替代关系
- [ ] 自动算料 (SPEC 3.8 P1)
  - **缺失**: 无法根据工单数量自动计算物料需求
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
- [ ] SN规则 (SPEC 4.8 P0)
  - **缺失**: 无产品序列号生成规则（CodeRule 未绑定 SN 生成）

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

### production-service 存在问题

- [ ] `ProductionDomainServiceImpl.startSchedule()` 返回 null — **未通过审查**：已修复但引入新问题
- [ ] `ProductionDomainServiceImpl.completeSchedule()` 返回 null — **未通过审查**：已修复但引入新问题

### digital-twin-service 集成

- [ ] mes-service `Equipment` 与 digital-twin-service `Device` 设备编码/ID 体系未统一

### 假代码/BUG 修正