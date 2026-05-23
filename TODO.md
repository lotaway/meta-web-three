# 以下清单在完成后需要确认勾选

- [ ] AI仓储功能（AI可选，回退则是无AI辅助建议，纯算法+人工处理）
- [ ] 是否完善了所必须的自动化测试

---

## 前端实现(apps/digital-twin)

### 1. 基础架构搭建
- [ ] 状态管理 (Redux/Zustand) - 实际未使用 Redux/Zustand，项目依赖 React 内置 useState/useContext
- [ ] 路由配置 - 实际无 react-router 等路由库，使用 window.location.hash + 条件渲染
- [ ] UI 组件库集成 (Ant Design) - 实际使用 styled-components，未集成 Ant Design

### GIS 地图模块 (可选)
- [ ] Cesium 地图集成
- [ ] 仓库布局可视化
- [ ] 物流路径展示

### 数据可视化
- [ ] ⚠️ DeviceChart.tsx:101 fillColor 函数仅 hex→rgba 和 rgb()→rgba() 正确，但 rgba() 输入会生成双重 alpha（如 `rgba(rgba(59,130,246,0.5), 0.3)`），3位hex被跳过。修复方向：解析前统一转为完整6位hex，或使用 Canvas 原生颜色解析

### 前端配置
- [ ] ⚠️ `VoiceTools.tsx` 和 `SubtitlesOverlay.tsx` 的 `API_BASE_URL` 中 host 仍硬编码为 `localhost`，仅端口可配。修复方向：引入统一配置模块，与 `digital-twin.ts` 相同模式支持 env 完整覆写（`AudioContext.tsx` 已通过 `VITE_VOICE_API_URL` 完整可配）

---

## 后端服务 (server/factory-domain/digital-twin-service)

### 1. 服务创建
- [ ] ❌ 主类未继承 `BaseApplication`，未使用 `@EnableDiscoveryClient`，不向 ZooKeeper 注册。修复方向：`DigitalTwinServiceApplication` 改为 `extends BaseApplication` + `@EnableDiscoveryClient`，Gateway 自动路由
- [ ] ❌ 未使用 `@EnableDubbo`，不支持 RPC。修复方向：添加 `@EnableDubbo` 注解，配置 Dubbo 扫描包路径

### 2. 领域模型
- [ ] ⚠️ 未使用值对象封装（`WorkshopId`、`Position`、`IPAddress` 等原始类型）

### 3. 持久化层
- [ ] `schema.sql` 已更新，新增 `alert_rules` 表。需执行 DDL 建表

### 4. 接口开发
- [ ] ❌ **完全无鉴权**：全控制器无 `@RequirePermission`，未读取 `X-User-Id`/`X-User-Role` 头。修复方向：注入 `@RequestHeader` 读取用户身份，引入 `@RequirePermission` 注解，参见下方"跨服务鉴权体系延伸"章节
- [ ] ❌ 无参数校验：全部使用 `Map<String, Object>` 而非 DTO，无 `@Valid`/`@Validated`。修复方向：为每个接口定义专用 DTO 类，添加 `@Valid`/`@Validated` 注解
- [ ] ❌ 无全局异常处理器（`@RestControllerAdvice`）。修复方向：创建 `@RestControllerAdvice` 类，统一处理 `MethodArgumentNotValidException`/`HttpMessageNotReadableException`/通用 `Exception`，返回标准错误响应体
- [ ] ❌ `valueOf(status.toUpperCase())` 传入非法值会抛出未捕获异常导致 500。修复方向：使用 `try-catch` 包裹 `valueOf`，非法值返回 400 或使用 safe default
- [ ] ❌ 无分页参数、无限流机制。修复方向：查询接口添加 `page`/`size` 参数，使用 MyBatis-Plus `Page` 分页，限制最大返回行数

### 5. 消息队列集成
- [ ] ❌ 无错误处理：Kafka 监听器无 try-catch，`broadcast()` 异常会导致消费失败。修复方向：在 `@KafkaListener` 方法内添加 try-catch，异常时记录日志 + 发送到错误 topic
- [ ] ❌ 无重试机制（`@RetryableTopic`）、无死信队列（DLT）。修复方向：使用 `@RetryableTopic` 注解配置重试次数和间隔，添加 DLT 消费者处理最终失败消息

### 6. WebSocket 服务
- [ ] ❌ **完全无认证**：`afterConnectionEstablished` 直接放行。修复方向：注入 `HandshakeInterceptor` 在握手阶段验证 token（query param 或自定义 header）
- [ ] ❌ **`setAllowedOrigins("*")`** 全源允许。修复方向：配置为具体前端域名列表，从 `application.yml` 读取
- [ ] ❌ 无心跳检测（ping/pong）。修复方向：实现 `WebSocketSession` 定时心跳（如每30s发送 ping），超时断开
- [ ] ❌ 无 `handleTransportError` 覆盖，传输异常不清理 session。修复方向：重写 `handleTransportError` 方法关闭并移除 session
- [ ] ❌ 使用 `System.out.println` 和 `e.printStackTrace()`。修复方向：替换为 `Logger`（如 SLF4J）
- [ ] ❌ 无 `@PreDestroy` 优雅关闭。修复方向：添加 `@PreDestroy` 方法，关闭所有 session 和线程池

## 数据库设计

### 1. PostgreSQL 表（MyBatis-Plus + schema.sql）
- [ ] ⚠️ 表名与 TODO 有差异（`devices` 非 `device_info`，`workshops` 非 `workshop_config` 等）
- [ ] ⚠️ CREATE TABLE 未指定字符集。修复方向：在 `schema.sql` 中为所有表添加 `DEFAULT CHARSET=utf8mb4`（兼容 MySQL），或使用数据库全局配置

## 联调测试

### 1. 单元测试
- [ ] 领域模型单元测试
- [ ] Service 层单元测试

### 2. 集成测试
- [ ] WebSocket 集成测试
- [ ] MQTT 消息集成测试
- [ ] Kafka 消费集成测试

### 3. 端到端测试
- [ ] 3D 场景加载测试
- [ ] 实时数据展示测试
- [ ] 告警流程测试

---

## 部署配置

### 1. Docker
- [ ] ⚠️ Dockerfile `ENTRYPOINT` 仍使用 `["sh", "-c", ...]` shell 形式，PID 1 为 sh 非 Java，`docker stop` 信号无法直达 JVM。修复方向：改用 exec 形式 `["java", "-jar", "app.jar"]`，JVM 参数通过 `JAVA_TOOL_OPTIONS` 环境变量传入
- [ ] ⚠️ Dockerfile 未利用层缓存：`COPY pom.xml .` 后未单独 `RUN mvn dependency:go-offline`，`COPY src` 变更仍会触发全部重新编译。修复方向：改为 `COPY pom.xml .` → `RUN mvn dependency:go-offline -B` → `COPY src ./src` → `RUN mvn clean package -DskipTests`

### 2. K8s
- [ ] ❌ K8s deployment 环境变量硬编码：secret 引用仅为注释示例，未实际激活（`imagePullPolicy: IfNotPresent` 已改）。修复方向：取消 secret 引用注释，结合 SealedSecret/External Secrets Operator 管理生产凭据

---

## 跨服务鉴权体系延伸（架构级，须统一处理）

Gateway（`server/gateway`）的 `UserAuthFilter` + JWT 鉴权体系原本只服务商城（mall-domain），现有 digital-twin、ERP、供应链等新域均未纳入。

### 当前状态

| 服务 | 注册到 ZooKeeper？ | Gateway 自动路由？ | 消费 `X-User-Id`/`X-User-Role` 头？ | 使用 `@RequirePermission`？ |
|------|-------------------|-------------------|--------------------------------------|----------------------------|
| `user-service` | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| `product-service` | ✅ 是 | ✅ 是 | ⚠️ 不消费但 Gateway 已验证 JWT | ⚠️ 无 |
| `order-service` | ✅ 是 | ✅ 是 | ⚠️ 同上 | ⚠️ 无 |
| **`digital-twin-service`** | ❌ 否（未继承 `BaseApplication`） | ❌ 否（ZooKeeper 不可发现） | ❌ 完全不读取 | ❌ 无 |
| **ERP 各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |
| **供应链各服务** | ❌ 否 | ❌ 否 | ❌ 完全不读取 | ❌ 无 |

### 需要做的工作（跨域统一方案）

1. **digital-twin-service**：主类继承 `BaseApplication` + `@EnableDiscoveryClient` + `@EnableDubbo`，注册到 ZooKeeper
2. **Gateway 白名单**：若 digital-twin WebSocket 端点无需鉴权，需加入 `excludedPathPatterns`；其余 API 通过 Gateway 统一鉴权
3. **Controller 改造**：注入 `@RequestHeader(HeaderConstants.USER_ID)` 和 `@RequestHeader(HeaderConstants.USER_ROLE)`，在业务方法中做授权决策
4. **`@RequirePermission` 拓展**：当前仅 admin 权限体系，需扩充到 digital-twin 的 device/workshop/alert 等资源（如 `dt:device:control`、`dt:alert:ack`）
5. **WebSocket 认证**：在握手阶段通过 `HandshakeInterceptor` 验证 token（query param 或 Sec-WebSocket-Protocol）
6. **ERP + 供应链同理**：统一规划权限资源树，避免各域各自造轮子

---

## 生产环境缺陷清单（待修复）

- [ ] ⚠️ DeviceChart.tsx:101 fillStyle 颜色转换 - fillColor 函数已实现 hex→rgba，但 rgba() 输入仍会生成双重 alpha，3位hex被跳过（与前端清单同一问题，需统一修复）
- [ ] ⚠️ 跨服务鉴权：digital-twin + ERP + 供应链均未纳入 Gateway 鉴权体系，见上方"跨服务鉴权体系延伸"章节
