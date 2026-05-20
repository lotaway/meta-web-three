# 以下清单在完成后需要确认勾选

- [] AI仓储功能（AI可选，回退则是无AI辅助建议，纯算法+人工处理）
- [] 所有docker和k8s运维配置文件是否都对应了当前全新的项目架构
- [] 是否完善了所必须的自动化测试

---

# 数字孪生 (Digital Twin) 实现清单

## 技术栈确认

- [x] 3D 可视化: Three.js + react-three/fiber + react-three/drei
- [x] GIS 地图: Cesium
- [x] 实时数据: WebSocket
- [x] IoT 设备: MQTT
- [x] 流数据: Kafka
- [x] 时序数据库: InfluxDB
- [x] 告警: Prometheus
- [x] AI 分析: Python
- [x] 设备连接: OPC-UA

---

## 前端实现(apps/digital-twin) [https://github.com/lotaway/meta-not](作为参考脚手架)

### 1. 基础架构搭建
- [ ] 项目初始化 (React + TypeScript)
- [ ] 状态管理 (Redux)
- [ ] 路由配置
- [ ] UI 组件库集成 (Ant Design)

### 2. 3D 场景模块
- [ ] Three.js 场景初始化
- [ ] 工厂车间 3D 模型加载
- [ ] 设备 3D 模型展示
- [ ] AGV/机器人 3D 模型动画
- [ ] 相机控制 (旋转、缩放、平移)
- [ ] 场景灯光与环境配置

### 3. 实时数据展示
- [ ] WebSocket 客户端配置
- [ ] 设备状态实时更新
- [ ] 生产线节拍可视化
- [ ] 产量/效率实时看板

### 4. GIS 地图模块 (可选)
- [ ] Cesium 地图集成
- [ ] 仓库布局可视化
- [ ] 物流路径展示

### 5. 数据可视化
- [ ] ECharts 图表集成
- [ ] 产量趋势图
- [ ] 设备 OEE 仪表盘
- [ ] 告警统计饼图

### 6. IoT 设备接入
- [ ] MQTT 客户端配置
- [ ] 设备状态主题订阅
- [ ] 传感器数据解析
- [ ] 设备告警推送

### 7. 告警管理
- [ ] 告警列表展示
- [ ] 告警规则配置
- [ ] 告警等级分类 (红/黄/蓝)
- [ ] 告警声音/弹窗通知

---

## 后端服务 (server/digital-twin-service)

### 1. 服务创建
- [ ] digital-twin-service 目录结构 (DDD 架构)
- [ ] pom.xml 依赖配置
- [ ] Application 启动类

### 2. 领域模型
- [ ] DeviceEntity - 设备实体
- [ ] WorkshopEntity - 车间实体
- [ ] ProductionLineEntity - 生产线实体
- [ ] AlertEntity - 告警实体

### 3. 接口开发
- [ ] 设备列表查询 API
- [ ] 设备实时状态 API
- [ ] 历史数据查询 API
- [ ] 告警记录 API

### 4. 消息队列集成
- [ ] Kafka Consumer 配置
- [ ] 设备状态消息消费
- [ ] 事件转发到 WebSocket

### 5. WebSocket 服务
- [ ] WebSocket 配置
- [ ] 客户端连接管理
- [ ] 实时数据推送

---

## 数据库设计

### 1. MySQL 表
- [ ] device_info - 设备信息表
- [ ] workshop_config - 车间配置表
- [ ] production_line - 生产线表
- [ ] alert_rule - 告警规则表
- [ ] alert_record - 告警记录表

### 2. InfluxDB (时序数据)
- [ ] device_metrics - 设备指标数据
- [ ] production_stats - 生产统计数据

---

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
- [ ] digital-twin-service/Dockerfile
- [ ] docker-compose.digital-twin.yml

### 2. K8s
- [ ] k8s/digital-twin-deployment.yaml
- [ ] k8s/digital-twin-service.yaml

---

## 优先级排序

### 第一阶段 (MVP)
1. 项目基础架构搭建
2. 3D 场景模块 (基础)
3. WebSocket 实时数据
4. 后端 API 开发
5. 设备状态展示

### 第二阶段 (增强)
1. IoT MQTT 接入
2. 数据可视化图表
3. 告警管理

### 第三阶段 (完善)
1. GIS 地图模块
2. AI 分析集成
3. OPC-UA 设备连接