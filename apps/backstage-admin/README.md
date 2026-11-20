# Backstage Admin

基于 Vue 3 + Element Plus 的后台管理系统，用于商品、订单、用户等业务模块管理。

## 技术栈

| 技术 | 用途 | 官网 |
| --- | --- | --- |
| Vue | 前端框架 | https://cn.vuejs.org/ |
| Element Plus | UI 组件库 | https://element-plus.org/ |
| Vue Router | 路由管理 | https://router.vuejs.org/ |
| Pinia | 状态管理 | https://pinia.vuejs.org/ |
| Pinia Plugin Persistedstate | Pinia 持久化 | https://prazdevs.github.io/pinia-plugin-persistedstate/ |
| Axios | HTTP 请求 | https://github.com/axios/axios |
| vue-echarts | 图表组件 | https://github.com/ecomfe/vue-echarts |
| TinyMCE Vue | 富文本编辑器 | https://github.com/tinymce/tinymce-vue |
| js-cookie | Cookie 管理 | https://github.com/js-cookie/js-cookie |

## 本地启动

1. 安装依赖

```bash
npm install
```

2. 配置环境变量

```bash
cp .env.example .env
```

`.env` 中的变量均为可选，未配置时使用 `src/utils/env.ts` 中的内置默认值：

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `VITE_BASE_SERVER_URL` | 后端网关地址（axios baseURL） | `http://localhost:10081` |
| `VITE_UPLOAD_URL` | 文件上传端点（相对网关） | `/platform-service/minio/upload` |
| `VITE_BLOCKCHAIN_ENABLED` | 区块链功能开关 | `false` |

> 其他设备（手机 / 局域网）访问时，把 `VITE_BASE_SERVER_URL` 配成后端所在机器的地址，
> 例如 `VITE_BASE_SERVER_URL=http://192.168.1.110:10081`。

3. 启动开发服务

```bash
npm run dev
```

4. 浏览器访问 [http://localhost:5173](http://localhost:5173)

## 后端网关路由约定

后端网关（`gateway`，端口 10081）开启了 Spring Cloud Gateway discovery locator，
所有请求需带 `/{serviceId}` 前缀，由网关按前缀路由到对应微服务，例如：

```
/mall-catalog-service/product/list
/mall-user-service/api/admin/user/list
/mall-trade-service/prefrenceArea/listAll
```

前端路径前缀与服务的对应关系维护在 `src/utils/http.ts` 的 `SERVICE_PREFIX_MAP`。
新增接口时必须同步维护该映射，否则请求会因为缺少服务前缀而返回 404。

## API Client

本应用的接口调用使用 `src/apis/` 下按业务模块组织的 **93 个手写模块**，
它们是所有视图的稳定公共接口，运行时不导入任何生成代码。

`yarn generate:api` 调用共享脚本 `apps/tools/OpenApiToTS.js`，从网关聚合文档
（`{VITE_BASE_SERVER_URL}/meta/v3/api-docs`）生成 TypeScript 客户端到
`src/generated/api/`。该目录**有意不被应用消费**——这是一处记录在案的有意偏离：
手写模块保留了更贴近业务的命名与封装，避免生成器重新运行时改动所有调用方。
生成结果仅作为 controller 签名与字段的参考。

生成器从 `.env` / `.env.local` 读取 `VITE_BASE_SERVER_URL` 与 `VITE_API_DOC_HOST`
（变量说明见 `.env.example`），两者缺省时回退到 `http://localhost:10081`。
`VITE_*` 前缀是 Vite 约定：只有以 `VITE_` 开头的变量才会注入 `import.meta.env`。

新增 / 变更接口时的同步机制：

1. 在对应 `src/apis/*.ts` 模块中手写接口函数，复用 `src/utils/http.ts` 的 `http`。
2. 若涉及新的网关服务前缀，同步更新 `src/utils/http.ts` 的 `SERVICE_PREFIX_MAP`。
3. 可选：运行 `yarn generate:api` 对照生成签名确认字段（运行需网关可访问）。
4. 保持既有函数签名兼容，避免改动调用方视图。

## 目录结构

```text
src
├── apis          # 按业务模块组织接口定义
├── assets        # 静态资源
├── components    # 可复用组件
├── icons         # SVG 图标
├── router        # 路由配置
├── store         # Pinia 状态定义
├── styles        # 全局样式
├── types         # 类型定义
├── utils         # 无业务状态的纯工具函数
└── views         # 页面模块
    ├── home
    ├── layout
    ├── normal
    ├── oms
    ├── pms
    ├── sms
    └── ums
```

## 开发约束（对齐 CODE_PRICEPLES）

- 命名必须表达业务语义，同一概念全项目只使用一个名称。
- 组件、状态、接口按模块拆分，避免单文件职责膨胀。
- 函数保持单一职责，优先使用防护语句，避免深层嵌套。
- 不依赖隐式共享状态，状态修改必须集中在明确位置。
- 禁止硬编码环境变量、路径、密钥和魔法数字，统一走配置。
- 输入必须显式校验，错误必须可定位，不允许吞异常。
- 公共接口结构保持稳定，变更必须兼容旧调用方。
- 核心业务逻辑必须配套单元测试，测试代码放在 `tests`（框架有强制约定时除外）。

## 提交前检查

建议在提交前至少执行：

```bash
npm run lint
npm run test
npm run build
```
