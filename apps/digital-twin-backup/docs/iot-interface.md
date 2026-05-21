# IoT 设备接入接口规范

## MQTT 主题规范

### 主题结构
```
{project}/{environment}/{device_type}/{device_id}/{data_type}
```

### 示例
```
meta-web-three/prod/sensor/warehouse-01/telemetry
meta-web-three/prod/camera/gate-02/event
meta-web-three/prod/actuator/robot-arm-03/command
```

## 设备类型定义

| 类型 | 用途 | 示例设备 |
|------|------|----------|
| sensor | 传感器 | 温湿度传感器、RFID读取器 |
| camera | 摄像头 | 监控摄像头、AGV摄像头 |
| actuator | 执行器 | 机械臂、自动门 |
| gateway | 网关 | 边缘计算网关 |
| agv | 自动化导引车 | 仓储AGV |

## 消息格式 (JSON)

### 1. 遥测数据 (telemetry)
```json
{
  "device_id": "sensor-001",
  "device_type": "sensor",
  "timestamp": "2026-05-20T14:00:00Z",
  "sequence": 1001,
  "values": {
    "temperature": 25.5,
    "humidity": 60.2,
    "battery": 85
  },
  "units": {
    "temperature": "°C",
    "humidity": "%",
    "battery": "%"
  }
}
```

### 2. 事件数据 (event)
```json
{
  "device_id": "camera-001",
  "device_type": "camera",
  "timestamp": "2026-05-20T14:00:01Z",
  "event_type": "motion_detected",
  "event_data": {
    "region": "A1",
    "confidence": 0.95,
    "image_url": "/storage/events/2026/05/20/camera-001-1002.jpg"
  }
}
```

### 3. 状态变更 (state)
```json
{
  "device_id": "agv-001",
  "device_type": "agv",
  "timestamp": "2026-05-20T14:00:02Z",
  "state": "idle",
  "previous_state": "moving",
  "location": {
    "x": 10.5,
    "y": 20.3,
    "z": 0,
    "zone": "warehouse-a"
  }
}
```

### 4. 命令响应 (command_response)
```json
{
  "device_id": "actuator-001",
  "device_type": "actuator",
  "timestamp": "2026-05-20T14:00:03Z",
  "command_id": "cmd-12345",
  "status": "success",
  "result": {
    "position": 180,
    "torque": 50
  }
}
```

## 设备注册接口

### REST API

#### 注册设备
```
POST /api/digital-twin/devices
```

Request:
```json
{
  "device_id": "sensor-001",
  "device_type": "sensor",
  "name": "仓库A温湿度传感器",
  "location": {
    "warehouse": "warehouse-a",
    "zone": "A1",
    "position": { "x": 10, "y": 5 }
  },
  "metadata": {
    "manufacturer": "Sensirion",
    "model": "SHT40",
    "firmware": "1.2.3"
  }
}
```

Response:
```json
{
  "device_id": "sensor-001",
  "status": "registered",
  "mqtt_topic": "meta-web-three/prod/sensor/sensor-001/#",
  "created_at": "2026-05-20T14:00:00Z"
}
```

#### 查询设备
```
GET /api/digital-twin/devices/{device_id}
```

#### 设备列表
```
GET /api/digital-twin/devices?type=sensor&zone=warehouse-a
```

## 实时数据订阅

### WebSocket 接口
```
ws://api.example.com/ws/digital-twin
```

订阅消息:
```json
{
  "action": "subscribe",
  "channels": [
    "device.sensor.*.telemetry",
    "device.agv.*.state"
  ]
}
```

推送消息:
```json
{
  "channel": "device.sensor.sensor-001.telemetry",
  "data": {
    "device_id": "sensor-001",
    "timestamp": "2026-05-20T14:00:00Z",
    "values": { "temperature": 25.5 }
  }
}
```

## 数据存储

### InfluxDB Measurement

```sql
-- 传感器数据
CREATE MEASUREMENT sensor_telemetry
TAGS: device_id, device_type, warehouse, zone
FIELDS: temperature, humidity, battery

-- 设备状态
CREATE MEASUREMENT device_state
TAGS: device_id, device_type, state
FIELDS: location_x, location_y, location_z

-- 事件记录
CREATE MEASUREMENT device_event
TAGS: device_id, device_type, event_type
FIELDS: confidence
```

## 接入流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   设备      │────▶│   MQTT      │────▶│   Kafka     │
│   上报      │     │   Broker    │     │   Topic     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Digital    │◀────│  消费服务   │
                    │  Twin App   │     │  (Worker)   │
                    └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  InfluxDB   │
                    │  存储       │
                    └─────────────┘
```

## 安全认证

### MQTT 认证
- Username: device_{device_id}
- Password: JWT token (设备注册时颁发)
- Topic 权限: 只能发布 own device topic

### API 认证
- Bearer Token: JWT
- 权限: digital-twin:read, digital-twin:write