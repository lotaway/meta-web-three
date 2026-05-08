微服务里“业务归属”优先于“管理端/用户端”维度，所以：

商品管理接口 → product-service
订单管理接口 → order-service
用户封禁接口 → user-service

而不是单独再做一个 admin-service 承载所有后台接口。

常见结构：

admin-web
    ↓
gateway
    ↓
各业务微服务

核心区别只是：

管理端调用的是 /admin/**
用户端调用的是 /api/**

但最终仍然路由到对应业务服务。

例如：

/admin/products
-> gateway
-> product-service

/admin/orders/refund
-> gateway
-> order-service

管理员鉴权通常也是放在网关第一层。

典型流程：

admin-web
-> JWT/Admin Token
-> gateway 校验
-> 转发微服务

网关负责：

token 校验
登录态解析
RBAC 权限校验
路由级权限
黑名单
限流
审计日志

然后把管理员信息透传给微服务：

X-Admin-Id
X-Admin-Role
X-Permissions

微服务内部通常不再重新解析 JWT，而是信任网关。

例如：

gateway:
  校验:
    admin:product:update

product-service:
  直接执行业务

但高安全操作通常仍会二次校验：

删除商品
资金操作
修改权限

避免有人绕过网关直接访问微服务。

因此内部还会：

校验 internal token
校验来源网关
mTLS
internal api key

否则：

curl product-service:8080/admin/deleteAll

就可能绕过 gateway。

实际企业里常见是：

外部权限:
  gateway

内部服务鉴权:
  spring security/resource server
  internal jwt
  mTLS

还有一种容易误解的点：

“后台管理端”不等于“管理员服务”。

管理员体系一般只单独拆：

iam-service
auth-service
rbac-service

负责：

管理员账号
角色
权限
登录
token

但具体业务管理接口仍在各业务服务里。

比较标准的结构：

gateway
auth-service
user-service
product-service
order-service
payment-service

其中：

/admin/users/**
-> user-service

/admin/products/**
-> product-service

而不是：

/admin/**
-> admin-service