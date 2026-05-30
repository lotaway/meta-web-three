# TODO

代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)

## 待完成任务

- [x] **tsconfig.json 弃用选项修复**：apps/digital-twin/system-management/tsconfig.json 中 moduleResolution=node10 已弃用（实际文件已使用 "Node"，问题已自行修复）
  - 错误信息：Option 'moduleResolution=node10' is deprecated and will stop functioning in TypeScript 7.0
  - 修复方式：在 tsconfig.json 的 compilerOptions 中添加 "ignoreDeprecations": "6.0"
  - 影响的文件：apps/digital-twin/system-management/tsconfig.json

- [x] **固定资产模块**：资产卡片、折旧计提（直线/双倍余额/年数总和）、资产盘点、资产减少/报废处置
- [x] **预算管理**：预算编制、执行控制、预算调整、预算与实际对比分析（后端实体/服务/Controller + 前端页面已完整实现）
- [x] **资金管理**：资金计划、现金流预测、银行对账、资金调拨（后端：实体/Repository/CommandService/QueryService/Controller 已完整实现；前端：仪表盘+5个子页面+API+路由已完整实现）
  - 实现功能：资金计划管理、银行账户管理、资金调拨管理、银行对账管理、现金流预测
  - 后端文件：
    - 实体：server/erp-domain/finance-service/src/main/java/com/metawebthree/finance/domain/entity/cash/*.java
    - Repository接口：server/erp-domain/finance-service/src/main/java/com/metawebthree/finance/domain/repository/cash/*.java
    - CommandService：server/erp-domain/finance-service/src/main/java/com/metawebthree/finance/application/command/cash/CashCommandService.java
    - QueryService：server/erp-domain/finance-service/src/main/java/com/metawebthree/finance/application/query/cash/CashQueryService.java
    - Controller：server/erp-domain/finance-service/src/main/java/com/metawebthree/finance/interfaces/facade/cash/CashController.java
    - DO/Mapper：server/erp-domain/finance-service/src/main/java/com/metawebthree/finance/infrastructure/persistence/dataobject/cash/*.java
  - 前端文件：
    - API：apps/backstage-admin/src/apis/cash.ts
    - 页面：apps/backstage-admin/src/views/cash/index.vue (仪表盘) + 5个子页面
    - 路由：apps/backstage-admin/src/router/index.ts
    - 国际化：apps/backstage-admin/src/locales/en-US.ts, zh-CN.ts
  - 偏差说明：后端Repository实现类（*RepositoryImpl.java）需要添加CashConverter转换器，当前使用了简化实现；部分细节功能（表单页面、详情页面）待完善
- [ ] **多币种核算**：支持外币凭证、汇兑损益处理
- [ ] **供应商绩效评估**：交货及时率、质量合格率、价格竞争力评分模型及看板
- [ ] **库存ABC分类管理**：按存货价值/周转率划分A/B/C类，制定不同管理策略
- [ ] **自动补货建议**：基于安全库存、历史销量、采购提前期自动生成请购单
- [ ] **需求预测**：基于时间序列或简单移动平均的销量预测
- [ ] **仓库质检记录**：独立质检标准表、质检结果（合格/不合格/让步接收）、不良品处理流程
- [ ] **仓库作业策略**：配置先进先出/后进先出/指定批次出库规则
- [ ] **物流费用自动结算**：物流服务与结算服务联动，自动生成运费结算单
- [ ] **供应商协同门户**：供应商自助查询订单、发货通知、对账
- [ ] **高级成本会计**：作业成本法（ABC）、标准成本与实际成本差异分析
- [ ] **财务比率仪表盘**：库存周转率、应收账款周转天数、应付账款周转天数、毛利率等关键指标可视化
- [ ] **报表订阅与自动发送**：定时生成报表并通过邮件/钉钉发送
- [ ] **人力资源（HRM）模块**：员工档案、组织架构、薪资核算、考勤管理（可选，后期规划）
- [ ] **项目管理模块**：项目预算、任务分解、工时填报、成本归集（可选，后期规划）
- [ ] 除了国际化文本和[本文档](TODO.md)可以包含中文外，其他所有文本内容都使用纯英文

## 已完成任务（审查后移除）

- ✅ 前端 TypeScript 编译错误修复：测试文件和生产代码类型错误已全部修复，vue-tsc 编译通过
- ✅ tsconfig.json 配置修复：baseUrl 已添加到 apps/backstage-admin/tsconfig.app.json