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

2. 启动开发服务

```bash
npm run dev
```

3. 浏览器访问 [http://localhost:5173](http://localhost:5173)

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
