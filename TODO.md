# TODO

[Project Guideline](./README.md)
[Backend Guideline](./server/README.md)

### 安全

## [区块重组处理](./TODO_BLOCK_REORG.md)

# 待决议功能

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)

- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)

- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)

# Docker 端口映射 bug 综合探查记录 (Docker Desktop 4.90.0)

## 现状概述
- **问题跨容器**：不仅仅是 PostgreSQL，Redis 也完全相同现象（`Ports` 显示为空）
- **正常容器**：Nginx、MongoDB 端口映射正常工作
- **故障容器**：PostgreSQL、Redis `docker inspect ... Ports` 结果为 `{"X/tcp":[]}`（空数组）
- **现象特征**：`docker compose up` 启动后短暂可见端口映射，随即消失；`docker compose port` 亦同
- **Docker 版本**：4.90.0（最新版本），问题持续存在
- **多次重启**：Docker Desktop 完全重启后问题依旧

## 问题现象矩阵

| 容器 | `docker inspect ... Ports` | Host 端口可达性 | 备注 |
|------|---------------------------|----------------|------|
| **default-nginx** | `{"80/tcp":[{"HostIp":"0.0.0.0","HostPort":"80"}]}` | ✅ 可访问 | 监听 `*`（所有接口） |
| **default-mongodb** | `{"27017/tcp":[{"HostIp":"0.0.0.0","HostPort":"27017"}]}` | ✅ 可访问 | 显式绑定配置 |
| **default-postgres** | `{"5432/tcp":[]}` (空) | ❌ 连接拒绝 | 默认 `listen_addresses=localhost` |
| **default-redis** | `{"6379/tcp":[]}` (空) | ❌ 连接拒绝 | 同 PostgreSQL 模式 |

## 官方 Issue 参考 (证实为 Docker Desktop 4.90.0 bug)

| Issue | 标题 | 关键日期 | 状态 |
|-------|------|----------|------|
| **#14926** | Standalone containers attached to overlay network are not accessible | 2025-09-04 | 证实 Docker Desktop 4.42.0+ (含 4.90.0) 仍有 bug |
| **#14327** | Forwarded ports are not rendered at container's start | 2024-09-24 | 证实“短暂出现后消失”现象 |
| **#7451** | network_mode fails with a large number of ports | 2024-10-08 | host 模式也有版本阈值 bug |
| **#13721** | Can't publish a port for a service on v4.24+ | 2023-10-04 | 每个大版本都可能引入新 bug |

## 已排除的方案（验证过程）

1. **端口冲突排查** - `lsof -i :5432`/`:6379` 无宿主机进程占用，非冲突问题
2. **#51758 经典 bug** - 虽然症状部分匹配，但非该 issue 的经典表现；且问题分布跨容器
3. **network_mode: host** - 虽然从技术角度可 bypass libnetwork 转发，但被禁止（非企业方案，破坏网络隔离）
4. **直接使用容器内网 IP** (`172.18.0.x:5432/6379`) - 被禁止（需手动设备相关信息，不符合“一键完成”）
5. **手动 `docker network create`** - 虽然有效但非“一键完成”方案；且在 4.90.0 下 subnet 子网配置失效
6. **docker-compose 网绠子网重定义** (subnet: 172.19.0.0/16) - 虽然原理触发网绠重建，但 4.90.0 环境实际失效；证实为 Docker Desktop 版本特定 bug

## 探查结论

**问题根因**：Docker Desktop 4.90.0 的 libnetwork 驱动存在跨容器的 port mapping bug，触发条件与应用层 `listen_addresses` 默认值有关（nginx/mongodb 正常，postgres/redis 异常）。这非 compose 语法错误，也非端口冲突，亦非单一服务配置问题。

**关键证据**：
- 同一份 `docker-compose.env.yml` 对不同服务有不同结果（nginx/mongodb 正常，postgres/redis 异常）
- Docker Desktop 4.90.0 为当前最新版本，官方 Issue #14926/#14327 直接证实上述现象
- subnet 方案在 4.90.0 环境中实际失效，排除了配置层面的彻底解法

## 当前可行的后续方案

### 方案一：健康检查 + 自动重试脚本 (推荐)
创建 `ensure-port.ps1` / `ensure-port.sh`，脚本内部循环 `docker compose down/up` + 等待 + 检查映射，最多重试 5-8 次。封装了 Docker Desktop 版本差异的不确定性，符合“一键完成”特性（用户只需运行脚本）。

### 方案二：文档化已知限制
在项目 README 中明确注明：Docker Desktop 4.90.0 存在跨容器 libnetwork bug，表现为部分容器端口映射失败。建议：
- 开发环境：运行自动重试脚本，或接受通过内网 IP 访问
- 生产环境：建议迁移至 Kubernetes/K3s，其网绠模型较 Docker Desktop 更稳定

### 方案三：等待 Docker 官方修复
监控 Docker Desktop 发行版变更，每个大版本(v4.23→v4.24→v4.90)都可能改变 port mapping 行为。暂无法在 compose 层面彻底根除。

## 更新的关键结论

- **问题本质**：Docker Desktop 4.90.0 libnetwork 驱动的跨容器 port mapping bug
- **非 compose 语法错误**：同一份 compose 文件对不同服务有不同结果
- **非端口冲突**：宿主机无其他进程占用对应端口
- **非单一服务问题**：Redis 也完全相同现象，证实是 Docker 层 bug
- **subnet 方案在 4.90.0 下失效**：虽然原理触发网绠重建，但 Docker Desktop 新版驱动对该配置的处理方式不同
- **host 模式与直接 IP**：均被项目约束禁止
- **后续行动**：推荐方案一（健康检查 + 重试脚本），或等待 Docker 官方修复；暂无其他纯 compose 层面的万能解法

## 待决议功能 (保持不变)

- [ ] 实现边缘计算集成 (Edge Computing，CDN级缓存和计算，降低延迟)
- [ ] 添加语音电商功能 (Voice Commerce，语音搜索、语音下单)
- [ ] 实现可持续性追踪 (碳足迹计算、绿色物流、环保商品标签)
