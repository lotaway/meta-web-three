# Digital Twin 技术栈规划

## 核心技术栈

### 3D 可视化
- **Three.js** - WebGL 3D 渲染引擎
- **@react-three/fiber** - React 绑定
- **@react-three/drei** - 常用组件库
- **@react-three/rapier** - 物理引擎

### 地理空间
- **CesiumJS** - 地理信息系统
- **Mapbox GL JS** - 2D 地图（可选）

### 实时通信
- **WebSocket** - 浏览器原生实时通信
- **socket.io** - WebSocket 封装库

### IoT 消息
- **MQTT.js** - MQTT 客户端
- **Eclipse Mosquitto** - MQTT Broker

### 事件流
- **Kafka** - 事件流处理（复用项目已有基础设施）
- **kafkajs** - Kafka Node.js 客户端

### 时序数据
- **InfluxDB** - 时序数据库
- **@influxdata/influxdb-client** - InfluxDB 客户端

## 项目依赖 (package.json)

```json
{
  "dependencies": {
    "three": "^0.160.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/drei": "^9.92.0",
    "@react-three/rapier": "^1.2.0",
    "cesium": "^1.110.0",
    "socket.io-client": "^4.7.0",
    "mqtt": "^5.3.0",
    "kafkajs": "^2.2.4",
    "@influxdata/influxdb-client": "^1.33.0"
  }
}
```

## 架构分层

```
┌─────────────────────────────────────────────┐
│              Presentation Layer             │
│  (Three.js Scene / Cesium Map / UI)         │
├─────────────────────────────────────────────┤
│              State Management               │
│  (Zustand / React Query)                    │
├─────────────────────────────────────────────┤
│              Data Layer                     │
│  ┌──────────┬──────────┬──────────┐        │
│  │ WebSocket│   MQTT   │  REST    │        │
│  └──────────┴──────────┴──────────┘        │
├─────────────────────────────────────────────┤
│              Infrastructure                 │
│  ┌──────────┬──────────┬──────────┐        │
│  │   Kafka  │ InfluxDB │  Redis   │        │
│  └──────────┴──────────┴──────────┘        │
└─────────────────────────────────────────────┘
```

## 核心模块

### 1. 场景管理 (SceneManager)
- 3D 场景初始化
- 相机控制
- 光照设置
- 资源加载

### 2. 设备接入 (DeviceConnector)
- MQTT 连接管理
- 设备状态同步
- 数据解析

### 3. 时序数据 (TimeSeriesStore)
- InfluxDB 写入
- 历史数据查询
- 聚合计算

### 4. 实时通信 (RealtimeChannel)
- WebSocket 管理
- 事件分发
- 心跳检测