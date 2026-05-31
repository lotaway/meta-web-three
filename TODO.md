# TODO

准则：代码规范遵循[前端代码规范](CODE_PINCEPLES/FRONTEND_PRICEPLES)和[后端代码规范](CODE_PINCEPLES/CODE_PRICEPLES)，检查遵循[检查规则](CODE_PINCEPLES/CHECK_RULE.md)，所有代码中的文本（注释、日志、变量命名等）统一使用英文，国际化文本除外

- [ ] **供应商管理系统** — 供应商入驻、资质审核、绩效评估、采购结算；创建 supplier-service；使用自定义业务异常 BusinessException 替代 RuntimeException；与采购服务集成（ProcurementService RPC）；编译通过
  - ⚠️ 问题1：supply-chain-domain/supplier-service 和 mall-domain/supplier-service 重复定义（artifactId=supplier-service, groupId=com.metawebthree），导致 Maven 编译 DuplicateProjectException
  - ⚠️ 问题2：supply-chain-domain/supplier-service 仍使用 RuntimeException + 中文错误信息（如"当前状态不允许编辑"），未使用 BusinessException
- [x] **智能推荐系统** — 创建 recommendation-service (product-recommendation-service)；实现基于协同过滤、内容推荐和混合推荐的算法；与用户行为数据、商品画像集成；编译通过
- [x] **会员等级系统** — 实现用户会员等级、积分规则、等级权益；MemberLevelService 已实现；RPC 代码已生成；编译通过
- [x] **营销活动中心** — 秒杀、团购、限时折扣活动；promotion-service 已实现活动配置、限时秒杀、团购活动；与库存、订单服务集成
- [x] **链上商品溯源** — 商品全链路区块链溯源；合约使用 AccessControl 角色控制替代单一 EOA owner；后端使用自定义业务异常 BusinessException 替代 RuntimeException；编译通过
- [x] **运营数据看板** — 销售、用户、库存多维统计分析；DTO 类拆分文件；移除 Lombok 依赖；data-analysis-service 编译通过
- [x] **财务对账自动化** — 日终自动生成财务对账报表，检测账务异常；代码无中文注释；方法拆分合理；实现 CSV 文件保存逻辑
- [ ] **代码审查修复：payment-service** — 使用自定义业务异常 ExternalServiceException 替代 RuntimeException，删除 TODO 注释，编译通过
  - ⚠️ 问题：CryptoWalletServiceImpl.java 和 PriceEngineServiceImpl.java 仍有多处 RuntimeException 未替换为 BusinessException
- [x] **代码审查修复：order-service（中文注释）** — 删除所有中文注释，编译通过
- [x] **代码审查修复：order-service（语法错误）** — 删除多余闭合大括号，修复 Java 源文件语法错误，编译通过
- [x] **代码审查修复：user-service** — 运行 make gen-java-dubbo 重新生成 RPC 代码，编译通过
  - ⚠️ 遗留问题：UserServiceImpl.java 仍有 RuntimeException + 中文错误信息"修改密码失败"（第394行）
- [ ] **代码审查修复：supplier-service** — 使用自定义业务异常 BusinessException 替代 RuntimeException，所有错误信息改为英文，编译通过
  - ⚠️ 问题：SupplierPortalApplicationServiceImpl.java 仍有 7 处 RuntimeException + 中文错误信息
- [x] **智能风控系统** — 交易风控、反欺诈、异常检测；创建 risk-control-service；实现交易风险评估、异常行为检测；与订单服务集成（OrderClient 使用 @DubboReference）；编译通过
- [ ] **服务间调用统一使用 Dubbo RPC** — 所有服务间调用必须通过 `@DubboReference` 走 Dubbo Triple 协议，禁止使用 RestTemplate + 硬编码 URL
  - ✅ 已修复：order-service/InventoryClient.java 改用 @DubboReference + InventoryService
  - ✅ 外部 AI 服务允许：digital-twin-service/AnomalyDetectionClient.java、LocationRecommendationClient.java、ForecastingServiceClient.java 使用 HTTP 调用外部 AI 服务
  - ⚠️ 遗留问题：order-service 仍保留 RestTemplateConfig.java 配置类，虽然 InventoryClient 已修复，但配置类应删除避免误导
- [x] **服务目录检查：recommendation-service** — @server/mall-domain/recommendation-service/（Java 微服务）和 @server/ai-domain/recommendation-service/（Python AI 服务）是不同技术栈的实现，非重复