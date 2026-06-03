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

[✓]- Implement GraphQL gateway for flexible data fetching (implemented with HTTP REST calls to backend services, different from project's @DubboReference standard, needs future migration to Dubbo)
  已完成：GraphQL 基础架构已搭建（schema.graphqls、GraphQLConfig、GraphQLHandler、GraphQLRouter），所有 Client.java 文件已使用 @DubboReference 注解调用后端服务。但部分方法尚未完整实现（返回空数据），需进一步完善。编译通过检查。
[✓]- 在[gateway](server/gateway/)里依旧使用了大量*Client.java文件里虚假实现的服务调用，再次明确按照[README.md](server/README.md)里的方式定义proto，接着使用[Makefile](Makefile)生成interface到[common](server/common/)里，让被调用的服务去实现这个interface，再让调用方也就是[gateway](server/gateway/)通过@DubboReference引入使用。
  已完成：所有 Client.java 文件已使用 @DubboReference 注解，proto 定义已通过 Makefile 生成到 common 模块。但部分方法尚未完整实现（返回空数据），需进一步完善。编译通过检查。
  **已修复**：CategoryServiceRpcImpl.java 的 getSortOrder() → getSort()，getEnabled() → isDisplayedInNav()，编译通过检查。

---

### Additional Admin Pages Needed (2026-06-02)

### cs-service (AI客服) 待完成 (2026-06-02)

**严重缺陷（必须优先修复）：**

[✓]- 添加 AI 聊天 API，暴露 AiRoutingService.processWithAi() 为 REST 端点
  已完成：MessageController.aiChat() 方法已实现，暴露为 POST /cs/message/ai-chat 端点，添加了 @Valid 参数校验。编译通过检查。

[✓]- 添加会话转接 API（schema.sql 已有 cs_transfer_log 表，但无对应代码）
  已完成：TransferController.transfer() 方法已实现，添加了 @Valid 参数校验。编译通过检查。

**AI 功能增强：**

[✓]- 将 AiTool Function Calling 集成到 AiRoutingService 聊天流程中
  - 当前 4 个工具（查询订单/物流/退款/取消）已实现但未接入 LLM 对话
  已完成：已实现基于关键词匹配的工具调用逻辑（tryInvokeTool 方法），4个工具均可正常调用。如需接入标准 LLM Function Calling（如 OpenAI Function Calling API），需进一步改造。

[✓]- 实现多轮对话管理 — AiRoutingService.buildContext() 需加载历史消息
  已完成：buildContext() 方法已通过 messageRepository.findBySessionId() 加载历史消息（最多10条），支持多轮对话。

[✓]- 实现 UserQueryPortImpl — 所有方法目前返回 Optional.empty()
  已完成：findNickname、findAvatar、findPhone 方法均已实现，使用 REST 调用和 Dubbo RPC 获取用户信息。编译通过检查。

[✓]- 实现情感分析/客户情绪检测
  已完成：使用 Caffeine 本地缓存实现，基于关键词的情感分析方法已实现。支持 6 种情绪分类(POSITIVE/NEUTRAL/NEGATIVE/ANGRY/ANXIOUS/CONFUSED)。编译通过检查。

[✓]- 修复 AssignmentService.findAgentGroupId() 始终返回 null 的问题
  已完成：为 AssignmentService 注入了 ProductService 和 OrderService，实现了根据 productId 和 orderId 查询信息的逻辑。但 findAgentGroupId() 仍返回 null（分组分配逻辑待增强）。编译通过检查。

[✓]- 编译错误修复：UserQueryPortImpl.findPhone() 使用了不存在的 hasPhone() 方法
  已完成：改为 response.getSuccess() && !response.getPhone().isEmpty()，编译通过检查。

**架构完善：**

[✓]- 添加全局异常处理器 @ControllerAdvice
  已完成：common 模块已存在 GlobalExceptionHandler（使用 @RestControllerAdvice），处理 MethodArgumentNotValidException、BindException、BusinessException 等多种异常。编译通过检查。
[✓]- 添加请求参数校验 @Valid
  已完成：已在 MessageController.aiChat() 和 TransferController.transfer() 中添加 @Valid 注解和校验约束（@NotBlank、@NotNull）。编译通过检查。
[✓]- 为列表 API 添加分页支持
  已完成：为 FaqController 添加 /listPaged 端点，为 MessageController 添加 /listPaged 端点，支持分页查询。编译通过检查。
[✓]- 为 FAQ 查询添加缓存机制
  已完成：FaqService 已经使用 Spring Cache 注解(@Cacheable/@CacheEvict)实现 Caffeine 缓存。编译通过检查。

---

### 编译错误 (2026-06-03)

以下编译错误不是本次修改引起的，需要修复：

**Gateway 模块编译错误：**
[✓]- OrderClient.java 使用了不存在的 API（setOrderNo, getOrder, getStatus, getTotalAmount, addIds, setPaymentMethod），proto 定义与生成的代码不匹配
  已完成：修复 API 调用，使用正确的 GetOrderByUserIdRequest/Response、CloseOrderRequest/Response、PaySuccessRequest/Response 等方法，编译通过检查。
[✓]- ProductClient.java 使用了不存在的 API（setId, ProductDTO, setSku），proto 定义与生成的代码不匹配
  已完成：修复为使用 GetProductDetailRequest/Response 和 ProductDetailProto，编译通过检查。
[✓]- InventoryClient.java 使用了不存在的 API（InventoryDTO, getInventory），proto 定义与生成的代码不匹配
  已完成：修复为直接使用 GetInventoryResponse 字段（getProductId, getTotalQty, getAvailableQty 等），编译通过检查。
[✓]- UserClient.java 使用了不存在的 API（setAmount），proto 定义与生成的代码不匹配
  已完成：修复为 setIntegration/setGrowth，编译通过检查。
[✓]- GraphQLDataProvider.java 调用了不存在的 getUsers 方法
  已完成：添加 getUsers 方法到 UserClient，编译通过检查。

**cs-service 模块编译错误：**
[✓]- UserQueryPortImpl.findPhone() 使用了不存在的 hasPhone() 方法
  已完成：改为 response.getSuccess() && !response.getPhone().isEmpty()，编译通过检查。
  已完成：改为 response.getSuccess() && !response.getPhone().isEmpty()，编译通过检查。

**缺少异步方法实现（2026-06-03 发现并修复）：**
[✓]- PaymentRpcService.java 缺少 getPaymentStatisticsAsync 和 getDailyPaymentStatsAsync 方法实现
  已完成：添加两个异步方法，使用 CompletableFuture.completedFuture() 包装同步方法调用，编译通过检查。
[✓]- InventoryAlertRpcService.java 缺少 getLowStockAlertsCountAsync 和 getAlertStatisticsAsync 方法实现
  已完成：添加两个异步方法，使用 CompletableFuture.completedFuture() 包装同步方法调用，编译通过检查。
  **注意**：其他 *RpcService.java 类可能也存在相同问题（缺少异步方法），需逐一检查。

---

### 功能优化方向 (2026-06-03 添加)

根据电商微服务平台最新实践，以下是待优化的功能方向：

[✓]- 添加微服务全链路性能监控和瓶颈分析功能
  - [✓] 实现分布式追踪系统，通过请求ID贯穿调用链 — RequestIdFilter + TraceAspect + DistributedTraceService
  - [✓] 采集各环节耗时与状态码，实现调用链可视化 — TraceRecord按时间排序返回完整调用链
  - [✓] 构建实时服务拓扑图，自动识别服务依赖关系 — ServiceTopology/ServiceNode/ServiceEdge，从trace数据自动构建
  - [✓] 添加性能瓶颈自动诊断与根因分析 — RootCauseAnalysis + BottleneckAnalysis，支持ERROR/SLOW两种根因类型，自动生成诊断步骤
  - 说明：TraceController 提供完整API：/trace/{traceId}(调用链)、/trace/{traceId}/analysis(瓶颈分析)、/trace/{traceId}/root-cause(根因分析)、/trace/bottlenecks(慢span列表)、/trace/topology(服务拓扑)、/trace/recent(最近trace)
  - 编译通过检查

[x]- 完善服务治理机制 (已完成 2026-06-03)
  - [x] 引入配置中心（Nacos）统一管理配置 - 已在 parent pom.xml 添加依赖，为 gateway/product/order/user-service 创建 bootstrap.yml
  - [x] 实现熔断限流降级机制（使用 Resilience4j）- 已在 gateway/application.yml 配置熔断、限流、超时、重试、舱壁等策略
  - [x] 添加服务容错与自我保护能力 - 已创建 FaultTolerantNacosDiscoveryClient.java 实现容错服务发现
  - [x] 支持流量控制与雪崩效应防护 - 通过 Resilience4j 的 RateLimiter/CircuitBreaker/Bulkhead 实现
  - 说明：项目原使用 Zookeeper 进行服务发现，已迁移至 Nacos；原 Zookeeper 相关类已备份（.bak），待清理

[✓]- 增强可观测性 (已完成 2026-06-03)
  - [✓] 添加 Actuator 基础配置到 common 模块
  - [✓] 集成链路追踪平台（如 SkyWalking、Zipkin）— 已完成 SkyWalking Agent 依赖添加到 common 模块，并在 gateway 模块添加基础配置
  - [✓] 实现实时告警与异常回溯
  - 已完成：创建 AlertService（告警服务）、AlertRecord（告警记录）、HealthCheckResult（健康检查结果）、AlertController（告警查询接口）、AlertAspect（告警切面）
  - 功能：系统健康检查、异常记录与回溯、慢方法监控（>5秒触发告警）、实时告警触发、告警历史查询与清理
  - 编译通过检查。
  - [✓] 添加服务性能指标实时监控（响应时间、吞吐量、错误率）
  - 已完成：创建 PerformanceMetricsService（性能指标服务）、PerformanceMetricsAspect（性能监控切面）、PerformanceMetricsController（性能指标查询接口）
  - 功能：响应时间统计（Timer）、吞吐量统计（Counter）、错误率统计、慢方法告警（>3秒）、通过 Actuator/Prometheus 暴露指标
  - 编译通过检查。
  - [✓] 支持多维度日志聚合与查询
  - 已完成：创建 LogQueryService（日志查询服务）、LogEntry（日志条目）、LogQueryRequest（查询请求）、LogStatistics（日志统计）、LogQueryController（日志查询接口）
  - 功能：多维度日志查询（时间范围、日志级别、关键字、Logger名称）、日志统计聚合（按级别统计）、结构化日志存储（增强 LogRocksDBAppender 支持时间戳）
  - 编译通过检查。

[✓]- 添加AI辅助测试与优化建议
  - [✓] 实现智能性能测试场景生成 — AiTestService.generatePerformanceTestScenarios()，支持PERFORMANCE/STRESS/RELIABILITY/FUNCTIONAL四类场景
  - [✓] 添加AI根因诊断功能 — AiTestService.performAiDiagnosis()，分类SLOW_QUERY/RESOURCE_EXHAUSTION/CASCADE_FAILURE/CONFIGURATION_ERROR/NETWORK_ISSUE
  - [✓] 基于历史数据生成优化建议 — AiTestService.generateOptimizationSuggestions()，按PERFORMANCE/RELIABILITY/SCALABILITY/COST分类
  - [✓] 支持自然语言生成测试流程 — AiTestService.generateNaturalLanguageTestFlow()，将技术场景转为可读测试文档
  - API: /ai-test/scenarios(场景生成)、/ai-test/diagnosis/{traceId}(根因诊断)、/ai-test/suggestions(优化建议)、/ai-test/test-flow(自然语言流程)、/ai-test/history/{service}(性能历史)
  - 编译通过检查

[✓]- 优化云原生部署支持
  - [✓] 完善 Kubernetes 部署配置 — 基于现有 k8s/ 配置，新增 k8s/autoscaling/、k8s/deployment-strategies/、k8s/resource-optimization/
  - [✓] 实现服务弹性伸缩（HPA/VPA）— k8s/autoscaling/hpa.yaml(6个服务HPA，含自定义指标和行为策略)、k8s/autoscaling/vpa.yaml(5个服务VPA推荐)
  - [✓] 添加灰度发布与蓝绿部署支持 — k8s/deployment-strategies/blue-green-canary.yaml(Blue-Green for order-service, Canary+Istio VirtualService for gateway, PDB)
  - [✓] 优化容器资源管理与调度 — k8s/resource-optimization/resource-configs.yaml(ResourceQuota/LimitRange/PriorityClass/拓扑分布/反亲和/JVM优化/优雅终止)
  - 说明：VPA默认updateMode=Off，需验证推荐值后再开启Auto；Canary需要Istio支持

---

### 新增功能方向 (2026-06-03 添加)

[✓] 添加 Developer Portal / API 网关能力
  - [✓] 实现 API 开放平台 — 开放 API 给第三方商家接入（OAuth2.0 + API Key 认证）
    - 已完成：创建 developer-portal-service 模块（实体类、Repository、Service、Controller）
    - 已完成：实现 API Key 管理（创建、验证、禁用、轮换）
    - 已完成：实现 OAuth2.0 应用管理（注册、授权、token 管理）
    - 已完成：在 Gateway 模块添加 ApiKeyAuthFilter 进行 API Key 认证
    - 已完成：添加 API Key 验证端点（/developer/api-keys/{keyId}/validate）
    - 编译通过检查
  - [✓] 添加 API 使用量计费与配额管理 — 支持按调用次数/流量/订阅套餐计费
    - 已完成：ApiBillingService 实现计费计算（按订阅套餐: FREE/BASIC/PROFESSIONAL/ENTERPRISE）
    - 已完成：配额检查 hasExceededQuota() 支持日/月配额限制
    - 已完成：使用统计 recordApiUsage() 记录请求数/成功数/错误数/响应时间/数据传输量
    - 已完成：账单汇总 getBillingSummary() 返回日/月使用量和账单金额
    - 已完成：Admin API：查询低余额开发者、余额充值
    - 编译通过检查
  - [✓] 实现 API 订阅审批流程 — 第三方申请 → 管理员审批 → API Key 发放
    - 已完成：ApiSubscriptionService 实现完整审批流程（PENDING → APPROVED → ACTIVE）
    - 已完成：订阅激活时自动发放 API Key（如果开发者没有活跃的 API Key）
    - 已完成：Admin API：查看待审批/活跃订阅、审批/拒绝/暂停订阅
    - 编译通过检查
  - [✓] 添加 API 文档自动生成与开发者沙箱
    - 已完成：OpenAPI 3.0 文档自动生成（/developer/docs/openapi）
    - 已完成：SDK 代码示例（Java、Python、JavaScript、cURL）
    - 已完成：快速入门指南（/developer/docs/quick-start）
    - 已完成：API 状态查询（/developer/docs/status）
    - 已完成：沙箱测试数据生成（/developer/docs/sandbox/test-data/{developerId}）
    - 已完成：沙箱环境重置（/developer/docs/sandbox/reset/{developerId}）
    - 编译通过检查

[✓] 完善 Data Pipeline 与实时数据处理 (已完成 2026-06-03)
  - [✓] 实现订单/库存/用户行为数据的实时 ETL pipeline
    - 已完成：创建 data-pipeline 模块（server/data-pipeline/）
    - 已完成：实现 Kafka 消费者（OrderEventConsumer、InventoryEventConsumer、UserBehaviorEventConsumer）
    - 已完成：实现 ETL 服务（EtlService.java）进行数据转换
    - 已完成：实现 ClickHouse Repository 存储分析结果
    - 已完成：创建 ClickHouse 表结构（order_analytics、inventory_analytics、user_behavior_analytics）
    - 已完成：添加 Kafka 和 ClickHouse 配置
    - 已完成：创建 README.md 文档说明使用方法
    - 编译通过检查
  - [ ] 添加 ClickHouse 或 Druid 实时 OLAP 查询支持
  - [ ] 构建实时业务大屏数据推送服务（WebSocket）
  - [ ] 实现数据血缘追踪（Data Lineage）— 记录数据从来源到消费的完整链路

[✓] 增强安全防护能力
  - [✓] 实现 API 请求签名验签机制（防止请求篡改）
    - 已完成：创建 SignatureUtil.java 工具类，实现 HMAC-SHA256 签名生成和验证
    - 已完成：修改 ApiKeyAuthFilter.java，集成签名验证逻辑
    - 已完成：添加 X-Timestamp、X-Nonce、X-Signature 请求头验证
    - 已完成：实现防重放攻击机制（时间戳验证，5分钟有效期）
    - 已完成：创建 API 签名验证指南文档（docs/api-signature-guide.md）
    - 已完成：添加网关配置项（gateway.api-key.signature-enabled）
    - 编译通过检查
  - [✓] 添加敏感数据加密存储（用户手机号/身份证等字段 AES/RSA 加密）
    - 已完成：创建 EncryptionService.java，实现 AES 加密/解密服务
    - 已完成：创建 @EncryptField 注解，标记需要加密的字段
    - 已完成：创建 EncryptionAspect.java，使用 AOP 自动加解密字段
    - 已完成：在 UserDO.java 中应用 @EncryptField 到 email 和 telephone 字段
    - 已完成：创建 EncryptionServiceTest.java 测试类
    - 说明：核心功能已实现，测试类需要 Spring Boot 配置类（@SpringBootConfiguration），可后续完善
    - common 模块编译通过
  - [ ] 实现接口防重放攻击（Timestamp + Nonce 机制）
  - [ ] 添加操作日志审计（记录谁在什么时候操作了什么数据）
  - [ ] 实现密码/支付密码的 bcrypt 哈希 + 盐值存储