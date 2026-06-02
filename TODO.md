# TODO

Guidelines: 
* Code should follow the [Frontend Code Principles](CODE_PINCEPLES/FRONTEND_PRICEPLES) and [Backend Code Principles](CODE_PINCEPLES/CODE_PRICEPLES), and be checked against the [Check Rules](CODE_PINCEPLES/CHECK_RULE.md). 
* All text in code (comments, logs, variable names, etc.) must use English uniformly, except for i18n text.
* After adding a backend service or feature, consider whether a corresponding admin page needs to be added to [backstage-admin](apps/backstage-admin/) or [digital-twin](apps/digital-twin/) or [Customer Client](apps/client/)

---

### [Backend Admin Missing]

The following backend services have been created, but  and  lack corresponding admin and operation pages. Each needs to be added:

- mall-domain (11 services, most missing admin pages)
- ai-domain (4 services)
- factory-domain / mes-service (production management admin)
- blockchain-domain (2 services)
- erp-domain (6 services: finance, HR, invoice, project, report, settlement)
- platform-domain (7 services: commission, customer service, data analysis, media, message, notification, user behavior)
- supply-chain-domain (6 services: inventory alert, inventory, logistics, procurement, supplier, warehouse)

---

### [Pending Features]

[]- Implement GraphQL gateway for flexible data fetching (implemented with HTTP REST calls to backend services, different from project's @DubboReference standard, needs future migration to Dubbo)

---

### Additional Admin Pages Needed (2026-06-02)

### cs-service (AI客服) 待完成 (2026-06-02)

**严重缺陷（必须优先修复）：**

[]- 添加 AI 聊天 API，暴露 AiRoutingService.processWithAi() 为 REST 端点
  问题：MessageController.aiChat()方法缺少@Valid参数校验，建议为AiChatRequest类添加javax.validation.constraints注解（如@NotBlank），并在方法参数中添加@Valid注解

[x]- 修复 MongoMessageRepository.findBySessionIdAfter() — 未按 afterMessageId 过滤（返回全量数据）
  状态：已通过检查，代码已包含afterMessageId过滤条件，编译通过

[]- 添加会话转接 API（schema.sql 已有 cs_transfer_log 表，但无对应代码）
  问题：TransferController.transfer()方法缺少@Valid参数校验，建议为TransferRequest类添加javax.validation.constraints注解，并在方法参数中添加@Valid注解

**AI 功能增强：**

[]- 将 AiTool Function Calling 集成到 AiRoutingService 聊天流程中
  - 当前 4 个工具（查询订单/物流/退款/取消）已实现但未接入 LLM 对话

[]- 实现多轮对话管理 — AiRoutingService.buildContext() 需加载历史消息

[]- 实现 UserQueryPortImpl — 所有方法目前返回 Optional.empty()

[]- 实现情感分析/客户情绪检测

[]- 修复 AssignmentService.findAgentGroupId() 始终返回 null 的问题

**架构完善：**

[]- 添加全局异常处理器 @ControllerAdvice
[]- 添加请求参数校验 @Valid
[]- 为列表 API 添加分页支持
[]- 为 FAQ 查询添加缓存机制
[]-[Gateway](server/gateway/)又是使用了大量RestTemplate进行服务间调用，这是错误的，应当使用