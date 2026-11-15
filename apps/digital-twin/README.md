# meta-note元笔记

AI 驱动的数字孪生与知识问答应用。

## 架构概览

```
apps/digital-twin/  ── Electron 桌面端
└── system-management   桌面客户端 (React + Electron + NestJS)

server/  ── Java 微服务后端
├── gateway                        统一入口（:10081，JWT 校验 + zk 服务发现）
└── factory-domain/factory-service  工厂服务（含数字孪生）(Spring Boot，不对宿主暴露端口)

客户端只能通过 gateway:10081/factory-service/** 访问 server/，禁止直连 factory-service。
```

## 目录说明

### `system-management/` — 桌面客户端

Electron 桌面应用，提供 AI 问答、语音交互、数字孪生 3D 场景。

| 功能 | 技术栈 | 状态 |
|------|--------|------|
| AI 问答（基于项目数据） | Local LLM Provider + RAG | 完成 |
| 语音 ASR/TTS | 麦克风捕获 + XTTS-v2 | 完成 |
| 3D 场景编辑器 | Three.js / @react-three/fiber | 完成 |
| **数字孪生工厂** | Three.js + REST API + WebSocket | **需 gateway + factory-service 配合** |

AI 问答对接 [Local LLM Provider](https://github.com/lotaway/local-llm-provider)，利用当前项目数据（文档、代码、知识库）进行检索增强生成（RAG）式回答。

**启动方式：**
```bash
cd system-management
yarn install
yarn dev            # 开发模式 (Vite + Electron)
```

### `system-support/` — Rust 原生模块

为桌面端提供音视频采集、编码和传输支持（N-API）。

```bash
cd system-support
cargo build --release
```

### `server/factory-domain/factory-service/` — 工厂服务（含数字孪生）

Spring Boot 微服务，提供数字孪生工厂 REST API + WebSocket 实时推送。

以下端点是 **factory-service 内部路径**；客户端一律通过 `http://localhost:10081/factory-service` 前缀访问，并携带 `Authorization: Bearer <GATEWAY_TOKEN>`。gateway 经 zk 解析服务后剥离 `/factory-service` 前缀再转发。

| 端点（gateway 前缀之后） | 说明 |
|------|------|
| `GET /api/digital-twin/devices` | 设备列表 |
| `GET /api/digital-twin/alerts/active` | 活跃告警 |
| `GET /api/digital-twin/stats/summary` | 统计摘要 |
| `POST /api/digital-twin/alert/{id}/acknowledge` | 确认告警 |
| `POST /api/digital-twin/alert/{id}/resolve` | 解决告警 |
| `GET /api/alert-rules`、`/api/mes/scada/*`、`/api/mes/trace/*` | 告警规则、SCADA 与追溯 |
| `ws://localhost:10081/factory-service/ws/digital-twin?token=<jwt>` | WebSocket 实时推送（握手 JWT 走 query，网关校验后剥离） |

WebSocket 事件：`DEVICE_STATUS_CHANGED`、`DEVICE_POSITION_UPDATED`、`PRODUCTION_OUTPUT_UPDATED`、`ALERT_CREATED`

**启动方式：**
```bash
cd ../../server
mvn clean package -pl factory-domain/factory-service -am -DskipTests
java -jar factory-domain/factory-service/target/factory-service-1.0.0.jar --spring.profiles.active=dev
```

### 子模块（待初始化）

- `browser-extension/` — 浏览器插件
- `CODE_PINCEPLES/` — 编码规范

## 技术栈

| 层 | 技术 |
|----|------|
| 桌面框架 | Electron 33 + Vite 5 |
| 前端渲染 | React 19 + TypeScript + Styled-Components |
| 3D 场景 | Three.js + @react-three/fiber + @react-three/drei |
| AI 推理 | [Local LLM Provider](https://github.com/lotaway/local-llm-provider) |
| 后端 (主进程) | NestJS 11 |
| 后端 (微服务) | Spring Boot 3 + Java 17 |
| 数据库 | SQLite (桌面端) / InfluxDB + Redis (服务端) |
| 消息队列 | Kafka |
| 原生模块 | Rust (napi-rs) |
| 构建工具 | Maven (Java) / Vite (前端) / Cargo (Rust) |
