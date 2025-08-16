# 支付/风控/对账/清结算功能

1. 支付（Payment）

## 核心概念

支付通道：微信、支付宝、银行卡、第三方聚合支付

订单生命周期：待支付 → 支付中 → 支付成功/失败 → 退款

幂等性：防止重复扣款（订单号唯一）

回调通知：异步/同步通知

签名验签：保证请求来源与数据完整性

安全加密：RSA、AES、HTTPS

## 开发步骤

统一支付接口设计

前端统一请求接口 /api/pay/create

支持多通道（策略模式）

生成支付订单

数据落库（订单号唯一）

记录通道信息和状态

调用第三方支付 API

请求参数加签

支付链接/二维码返回给前端

接收支付回调

验签 → 幂等处理 → 更新状态

主动查询支付状态

防止漏单（轮询第三方）

## 测试方式

沙箱环境（支付宝、微信）

模拟回调请求（Postman）

并发压测幂等逻辑（JMeter）

Java 示例（策略模式实现多通道支付）

```java
public interface PayService {
    PayResponse pay(PayRequest request);
}

public class WechatPayService implements PayService {
    public PayResponse pay(PayRequest request) {
        return new PayResponse("wechat_qr_code_url");
    }
}

public class AlipayService implements PayService {
    public PayResponse pay(PayRequest request) {
        return new PayResponse("alipay_qr_code_url");
    }
}

public class PayServiceFactory {
    public static PayService getService(String channel) {
        if ("WECHAT".equals(channel)) return new WechatPayService();
        if ("ALIPAY".equals(channel)) return new AlipayService();
        throw new IllegalArgumentException("unsupported channel");
    }
}
```

2. 风控（Risk Control）

## 核心概念

实时风控：交易发生时立即判断

规则引擎：金额、频率、地理位置、黑名单

设备指纹：识别唯一设备

机器学习反欺诈：可选

## 开发步骤

数据采集

用户 IP、设备信息、交易金额、历史记录

规则配置

规则表（金额上限、频率限制、地域限制）

风控引擎执行

规则匹配 → 拦截 / 放行

告警与人工审核

高风险单进入人工审核池

## 测试方式

模拟多笔相同 IP/卡号交易

金额超限交易

异地登录 + 支付

Java 示例（简单规则引擎）

```java
public class RiskRule {
    public boolean check(PayRequest req) {
        if (req.getAmount() > 10000) return false;
        if ("192.168.1.1".equals(req.getIp())) return false;
        return true;
    }
}
```

3. 对账（Reconciliation）

## 核心概念

内部账：本系统订单数据

外部账：第三方支付平台账单

对账差异：漏单、多单、金额不一致

账务调整：退款、补单

## 开发步骤

拉取第三方账单文件

定时任务（每日凌晨）

解析账单

CSV/Excel 解析

对比内部账与外部账

按订单号/金额/状态匹配

生成差异报告

分类（少记、多记、金额不符）

处理差异

人工核实 → 数据调整

## 测试方式

造测试账单文件

造测试账单文件

内部造一笔缺失订单

模拟金额差异

Java 示例（账单对比）

```java
public class Reconciliation {
    public List<String> diff(List<Order> internal, List<Order> external) {
        Set<String> internalIds = internal.stream().map(Order::getId).collect(Collectors.toSet());
        Set<String> externalIds = external.stream().map(Order::getId).collect(Collectors.toSet());
        return externalIds.stream().filter(id -> !internalIds.contains(id)).collect(Collectors.toList());
    }
}
```

4. 清结算（Clearing & Settlement）

## 核心概念

清分：将交易按商户、通道、时间维度汇总

结算：将清分后的金额打款给商户

T+N 结算：交易日后 N 天结算

手续费：扣除手续费后的结算金额

## 开发步骤

清分

定时统计交易数据（按商户/通道）

结算单生成

计算应结金额 = 总金额 - 手续费

打款

自动银行转账/人工打款

结算对账

确认打款与结算单一致

## 测试方式

不同商户/通道组合

手续费计算验证

模拟跨月、节假日结算

Java 示例（简单清分计算）

```java
public class Settlement {
    public BigDecimal calculate(BigDecimal total, BigDecimal feeRate) {
        return total.subtract(total.multiply(feeRate));
    }
}
```
