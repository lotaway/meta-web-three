# TODO

准则：代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)，所有代码中的文本（注释、日志、变量命名等）统一使用英文，国际化文本除外
添加后端服务与功能后，需要考虑是否需要添加浏览与操作到[后台管理](apps/backstage-admin/)或[数字孪生](apps/digital-twin/)里

---
**【待规划功能】（2026-06-01评估）**

（暂无已完成项目）

**【前端管理后台缺失】：** 

以下后端服务已创建，但 `apps/backstage-admin/` 和 `apps/digital-twin/` 中缺少对应的管理和操作页面，需逐项补充：

- mall-domain（11 个服务中大部分缺少后台管理页面）
- ai-domain（4 个服务）
- factory-domain / mes-service（生产管理后台）
- blockchain-domain（2 个服务）
- erp-domain（6 个服务：财务、HR、发票、项目、报表、结算）
- platform-domain（7 个服务：佣金、客服、数据分析、媒体、消息、通知、用户行为）
- supply-chain-domain（6 个服务：库存预警、库存、物流、采购、供应商、仓库）

**本次迭代完成：**
- HR管理后台（erp-domain/hrm-service）：已创建HR管理页面，包含部门管理（树形结构、增删改查）和员工管理（列表、新增、编辑、删除、按部门筛选）功能，TypeScript编译通过，代码规范检查已通过
- 仓库管理后台（supply-chain-domain/warehouse-service）：已创建 warehouse 管理页面，包含仓库管理、入库管理、出库管理、库存管理功能，TypeScript 编译通过，代码规范检查已通过

---
**【代码规范审查结论】（2026-06-01）**

本次审查处理结果：
- 智能客服系统架构设计：已完成代码规范检查和编译验证通过（WorkOrder、Faq领域模型设计，WorkOrderRepository、FaqRepository接口，WorkOrderService、FaqService、TicketClassificationService应用服务，AI工单智能分类实现），已通过检查并从清单中移除
- 供应商管理系统：artifactId 修复（mall-supplier-service）已通过检查并从清单中移除
- 营销活动中心：PromotionException 继承 BusinessException 已通过检查并从清单中移除
- 链上商品溯源：使用 OpenZeppelin initializer 修饰符，符合区块链合约安全最佳实践，已通过检查并从清单中移除
- 编译阻塞已修复：OrderPaymentSaga.java 移除 @Data 注解，order-service 编译成功，已通过检查并从清单中移除
- mall-supplier-service服务目录名：已通过检查并从清单中移除
- 推荐服务架构迁移：编译通过，代码规范问题已修复，已通过检查并从清单中移除
- 智能推荐系统：代码已实现基于配置的真算法逻辑，编译通过，已通过检查并从清单中移除
- 会员等级系统：编译通过，已通过检查并从清单中移除
- 运营数据看板：编译通过，已通过检查并从清单中移除
- 财务对账自动化：编译通过，已通过检查并从清单中移除
- 智能风控系统：编译通过，已通过检查并从清单中移除
- Solidity 编译错误（GoodsNFT.sol、CommitReveal.sol）- 已修复，编译通过
- 售后服务系统、商品评价系统、库存预警系统、智能客服系统：编译通过，代码规范检查已通过，从清单中移除
- 消息通知中心：代码已实现真实 RPC 调用和短信网关集成，编译通过，代码规范问题已修复，已从清单中移除
- 库存预警系统管理后台：编译通过，代码实现完整，已通过检查并从清单中移除
- 推荐系统管理后台：编译通过，代码实现完整（deleteRuleAPI已实现，handleDelete调用真实API，后端deleteRule端点存在），已通过检查并从清单中移除
- 前端项目编译错误已修复（http.ts 中 http.get/post/put/delete 方法类型定义完整，index.vue 中 TypeScript 类型错误已修正），编译通过，已通过检查并从清单中移除
- 直播带货系统架构设计：LiveApplicationService 方法已拆分为私有辅助方法，符合单函数不超过20行规范；已新增 LiveApplicationServiceTest 单元测试类（9个测试用例覆盖核心业务逻辑）；编译通过，已通过检查并从清单中移除
- 社交电商系统架构设计：编译通过，代码规范检查已通过，已从清单中移除
- 添加商品管理后台：编译通过，代码实现完整，已通过检查并从清单中移除
- 添加物流管理后台：TypeScript编译错误已修复，编译通过，代码实现完整，已通过检查并从清单中移除
- 添加 AI 服务管理后台：TypeScript 编译错误已修复（StatusTagType 类型定义修复），编译通过，代码规范检查已通过（forecasting、route-optimizer 页面完整实现），已通过检查并从清单中移除
- 添加区块链服务管理后台：TypeScript 编译错误已修复（图标 Truck/Package 替换为 Van/Box，StatusTagType 类型修复），编译通过，代码规范检查已通过（traceability、wallet 页面完整实现），已通过检查并从清单中移除
- 添加供应商管理后台：代码实现完整，后端 supplier-service 实现真实业务逻辑（SupplierController、SupplierApplicationServiceImpl），前端 backstage-admin 供应商管理页面功能完整（列表、新增、编辑、审核、详情），编译通过，代码规范检查已通过，从清单中移除
- 添加支付管理后台：TypeScript编译通过，后端 payment-service 实现真实业务逻辑（PayController、ExchangeOrderController、ReconciliationReportController），前端 backstage-admin 支付管理页面功能完整（支付订单列表、退款管理、支付配置、对账报表），代码规范检查已通过，已通过检查并从清单中移除
- 添加商品评价管理后台：后端 review-service 实现真实业务逻辑（ReviewApplicationService、ReviewController），前端 backstage-admin 商品评价管理页面功能完整（评价列表、回复、审核、详情），编译通过，代码规范检查已通过，从清单中移除

**本次审查移除项目：**
- 智能客服系统：后端 cs-service 编译通过，代码实现真实业务逻辑（ChatCompletionRequest/Response DTO、AI工具类、WorkOrder/Faq领域模型、ConversationService/AgentService/MessageService/AiRoutingService应用服务），前端 cs 管理页面存在，代码规范检查已通过，已从待规划功能清单中移除
- 直播带货系统：后端 live-service 编译通过，代码实现真实业务逻辑（LiveApplicationService、AnchorRepository、LiveRoomRepository、LiveCommentRepository、LiveOrderRepository等），前端 live 管理页面存在，代码规范检查已通过，已从待规划功能清单中移除
- 社交电商系统：后端 social-commerce-service 和 group-buying-service 编译通过，代码实现真实业务逻辑（ShareRewardConfig、DistributionRelation、Community等领域模型，SocialCommerceApplicationService、GroupBuyApplicationService应用服务），代码规范检查已通过，已从待规划功能清单中移除
- 直播带货管理后台：前端 backstage-admin 直播管理页面功能完整（直播间管理、主播管理、商品挂载、弹幕管理、订单管理），TypeScript 编译通过，代码规范检查已通过，从清单中移除
- 社交电商管理后台：前端 backstage-admin 社交电商管理页面功能完整（拼团活动、分享奖励、分销管理、社群管理、订单管理），创建 social.ts API 文件，TypeScript 编译通过，代码规范检查已通过，从清单中移除

**修正需求：**
- Solidity 编译错误（GoodsNFT.sol、CommitReveal.sol）- 已修复
  - GoodsNFT.sol: 添加 COMMISSION_RATE 和 UPLINE_COMMISSION_RATE 常量定义，修复 finalPrice 未声明问题，添加 discount 计算逻辑
  - CommitReveal.sol: 修复变量遮蔽警告（commit 变量重命名为 userCommit）
  - hardhat.config.js: 添加 viaIR: true 解决 "Stack too deep" 编译错误

**备注**：
- 推荐系统管理后台：已从清单中移除，代码符合规范要求
- 库存预警系统管理后台：已从清单中移除，代码符合规范要求
- 前端项目编译错误已修复，编译通过
- 直播带货系统架构设计：已从清单中移除，代码符合规范要求
- 社交电商系统架构设计：已从清单中移除，编译通过，代码规范检查已通过
- 合并 notification-service 到 message-service：已从清单中移除，编译通过，代码规范检查已通过
- 添加商品管理后台：已从清单中移除，编译通过，代码规范检查已通过
- 添加物流管理后台：已从清单中移除，编译通过，代码规范检查已通过
- 添加供应商管理后台：已从清单中移除，编译通过，代码规范检查已通过
- 添加支付管理后台：已从清单中移除，编译通过，代码规范检查已通过
- 添加商品评价管理后台：已从清单中移除，编译通过，代码规范检查已通过