# backstage-admin

### 技术选型

| 技术                        | 说明                  | 官网                                                    |
| --------------------------- | --------------------- | ------------------------------------------------------- |
| Vue                         | 前端框架              | https://cn.vuejs.org/                                   |
| Element Plus                | 前端UI框架            | https://element-plus.org/                               |
| Vue Router                  | 路由框架              | https://router.vuejs.org/                               |
| Pinia                       | 全局状态管理框架      | https://pinia.vuejs.org/                                |
| Pinia Plugin Persistedstate | Pinia持久化插件       | https://prazdevs.github.io/pinia-plugin-persistedstate/ |
| Axios                       | 前端HTTP框架          | https://github.com/axios/axios                          |
| vue-charts                  | 基于Echarts的图表框架 | https://github.com/ecomfe/vue-echarts                   |
| TinyMCE Vue                 | 富文本编辑器          | https://github.com/tinymce/tinymce-vue                  |
| Js-cookie                   | cookie管理工具        | https://github.com/js-cookie/js-cookie                  |
| vue-element-admin           | 项目脚手架            | https://github.com/PanJiaChen/vue-element-admin         |

### 项目布局

```lua
src -- 源码目录
├── apis -- axios网络请求接口定义
├── assets -- 静态图片资源文件
├── components -- 通用组件封装
├── icons -- svg矢量图片文件
├── router -- vue-router路由配置
├── store -- pinia的状态管理
├── styles -- 全局css样式
├── types -- 类型定义
├── utils -- 工具类
└── views -- 前端页面
    ├── home -- 首页
    ├── layout -- 通用页面框架
    ├── normal -- 常用页面（login和404）
    ├── oms -- 订单模块页面
    ├── pms -- 商品模块页面
    ├── sms -- 商品模块页面
    └── ums -- 用户模块页面
```

## 搭建步骤

- 在命令行中运行命令：`npm install`,`npm run dev`,运行项目;
- 访问地址：[http://localhost:5173](http://localhost:5173)即可打开后台管理系统页面;