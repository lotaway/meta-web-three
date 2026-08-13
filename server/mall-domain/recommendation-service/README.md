# recommendation-service（推荐 + AI 辅助购物）

基于 zk+dubbo+spring-cloud-gateway 的推荐与 AI 辅助购物服务，端口 `10104`。

## 功能

- **商品推荐**：规则 / 算法生成个性化推荐（已有功能，未改动）。
- **AI 辅助购物**：
  - 文本纠错：LLM 纠错（OpenAI 兼容 `/v1/chat/completions`），失败自动降级为本地商品词典纠错（bigram + 编辑距离）。
  - 智能匹配：文本向量化（`/v1/embeddings`）+ 向量库检索。
  - 以图搜图：图片向量化（`/v1/images/embeddings`）+ 向量库检索。
  - 一站式搜索：`纠错 → 智能匹配` 聚合。

## AI 辅助购物 API

> 路由约定：所有 REST 请求以 **gateway 为入口、配合 ZK 服务发现** 转发（Spring Cloud Gateway discovery locator）。
> C 端/后台调用 URL 需带服务名前缀 `/recommendation-service/...`，网关自动 `StripPrefix(1)` 后转发到本服务。
>
> - C 端客户端（apps/client）调用：`{gateway}/recommendation-service/api/ai-shopping/...`
> - 后台（backstage-admin）：`http` 工具根据 `SERVICE_PREFIX_MAP` 自动为 `/api/admin/ai-shopping/*` 添加 `recommendation-service` 前缀

### C 端（需登录）
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ai-shopping/text-correct` | 文本纠错，body `{"text":"..."}` |
| POST | `/api/ai-shopping/smart-match` | 智能匹配，body `{"query":"...","topK":n}` |
| POST | `/api/ai-shopping/image-search` | 以图搜图，multipart `image` + 可选 `topK` |
| POST | `/api/ai-shopping/search` | 一站式：`{"q"/"query":"...","topK":n,"userId":id}` |

### 后台管理（backstage-admin）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/admin/ai-shopping/config` | 配置项列表 |
| POST | `/api/admin/ai-shopping/config` | 新增/覆盖配置 `{configKey,configValue,description}` |
| DELETE | `/api/admin/ai-shopping/config/{key}` | 删除配置 |
| POST | `/api/admin/ai-shopping/index/rebuild?type=all\|text\|image` | 重建向量索引 |
| GET | `/api/admin/ai-shopping/index/status` | 索引状态 |
| POST | `/api/admin/ai-shopping/provider/test?type=embedding\|image\|llm` | Provider 连通性测试 |
| GET | `/api/admin/ai-shopping/logs?limit=50` | 最近搜索日志 |

### GraphQL（网关 federation）
`aiTextCorrect(text:)`、`aiSmartMatch(query:, topK:)` 已注册到网关 `FederationRouter`。

## 配置

配置层级：`application.yml` 默认值 < `ai_shopping_config` 表（DB 覆盖，优先级更高，后台页面可维护）。

### application.yml（`ai-shopping.*`）

```yaml
ai-shopping:
  enabled: true
  vector-store: memory            # milvus | memory（内存兜底，余弦扫描）
  embedding-dim: 1024
  default-top-k: 20
  embedding:            # 文本嵌入（OpenAI 兼容 /v1/embeddings）
    base-url: ""        # 必填，如 https://api.openai.com/v1
    api-key: ""
    model: ""
    path: ""            # 自定义请求路径，默认 /embeddings
    timeout-ms: 15000
    max-retries: 2
  image-embedding:      # 图像嵌入（OpenAI 兼容 /v1/images/embeddings）
    base-url: ""
    api-key: ""
    model: ""
    path: ""
    timeout-ms: 15000
    max-retries: 2
  llm:                  # 文本纠错 LLM（OpenAI 兼容 /v1/chat/completions）
    base-url: ""        # 必填，如 https://api.openai.com/v1
    api-key: ""
    model: ""           # 如 gpt-4o-mini / qwen-max
    path: ""
    timeout-ms: 15000
    max-retries: 2
  milvus:
    host: localhost
    port: 19530
    api-token: ""
    collection-text: product_text
    collection-image: product_image
```

> `base-url` / `api-key` / `model` 留空时 LLM / 向量化功能不可用：文本纠错自动降级为本地词典纠错，
> 智能匹配/以图搜图会因缺少向量化而报错（可在后台「Provider 测试」确认配置是否生效）。

### DB 覆盖（`ai_shopping_config` 的 `config_key`）
`embedding.base-url` / `api-key` / `model`、`image-embedding.*`、`llm.*`、`vector.store`、
`milvus.host` / `milvus.port` / `milvus.collection-text` / `milvus.collection-image`。

## Milvus 部署（网络已统一）

`docker-compose.ai.yml` 已纳入 `docker-compose.yml` 的 `include`，etcd/minio/milvus 统一挂载到
`meta-web-three` 网络，与后端服务同网互访（不再使用独立的 `default-ai` 网络）。

- Milvus 2.6.5，端口 `19530`（gRPC/HTTP REST）/ `9091`（健康检查）。
- 服务间通过容器名互访：`milvus:19530`；`docker-compose.server.yml` 已为 recommendation-service
  注入 `AI_SHOPPING_MILVUS_HOST=milvus`。
- 本地开发（`run-server.sh`，非 Docker）：milvus 跑在宿主机时用默认 `localhost`，或用环境变量
  `AI_SHOPPING_MILVUS_HOST` / 后台 DB 配置覆盖。
- 向量库通过 Milvus REST API（`/v2/vectordb/...`，JDK HttpClient）交互，未引入 milvus-sdk-java；
  `vector.store=memory` 时使用内存向量库兜底。

## 数据表

`schema.sql`：`ai_shopping_config`（运行时配置覆盖）、`ai_search_log`（搜索日志）。

## 单元测试

```bash
# 需先 install 父 POM / common / event-sdk
mvn -N install -DskipTests && mvn -pl common install -DskipTests
cd ../shared/event-sdk && mvn install -DskipTests

mvn -pl mall-domain/recommendation-service test
```
