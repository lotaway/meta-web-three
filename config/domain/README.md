# 领域边界配置

本目录包含各服务的领域边界配置文件。每个 YAML 文件定义一个领域及其所属服务的允许访问表。

## 文件结构

```
config/domain/
├── README.md                    # 本文件
├── mall-domain.yaml             # 商城域
├── supply-chain-domain.yaml     # 供应链域
├── factory-domain.yaml          # 工厂域
├── ai-domain.yaml               # AI 域
├── platform-domain.yaml         # 平台域
└── blockchain-domain.yaml       # 区块链域
```

## 配置格式

```yaml
owner_domain: <领域名>
description: <领域描述>
services:
  <服务名>:
    allowed_tables:
      - <允许访问的表>
    readonly_tables:
      - <只读表>
```

## 使用方法

### 本地检查

```bash
# 检查特定服务
./scripts/check-domain-boundary.sh order-service

# 检查当前目录所在的服务
cd server/mall-domain/order-service
../../scripts/check-domain-boundary.sh
```

### CI 自动检查

GitHub Actions 会自动在每次 push 和 PR 时运行检查。

## 添加新服务

1. 找到对应领域的 YAML 文件
2. 在 `services` 下添加新服务配置
3. 定义 `allowed_tables` 和 `readonly_tables`

## 禁止事项

- ❌ 禁止直接修改非本领域的表
- ❌ 禁止跨域事务
- ❌ 禁止通过同步 API 调用修改其他领域数据

## 推荐方式

- ✅ 通过事件驱动与其他领域通信
- ✅ 通过 API 查询其他领域数据（只读）
- ✅ 通过事件订阅实现数据一致性