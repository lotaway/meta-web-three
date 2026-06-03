# API 请求签名验证指南

## 概述

为了防止 API 请求被篡改，meta-web-three 项目实现了请求签名验证机制。所有第三方开发者调用开放 API 时，必须在请求中包含签名信息。

## 签名原理

1. **签名生成**：客户端使用 HMAC-SHA256 算法，根据请求信息生成签名
2. **签名验证**：服务端使用相同算法验证签名，确保请求在传输过程中未被篡改
3. **防重放攻击**：通过时间戳和随机数（nonce）防止请求被重复执行

## 请求头要求

所有需要签名的 API 请求必须包含以下请求头：

| 请求头 | 说明 | 示例 |
|--------|------|------|
| `X-API-Key` | 开发者的 API Key ID | `key_1234567890abcdef` |
| `X-API-Secret` | API Key 的密钥（用于签名生成） | `secret_abcdef1234567890` |
| `X-Timestamp` | 请求时间戳（秒级） | `1717459200` |
| `X-Nonce` | 随机字符串（16位） | `a1b2c3d4e5f6g7h8` |
| `X-Signature` | 请求签名 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## 签名生成步骤

### 1. 收集请求信息

- **HTTP 方法**：GET, POST, PUT, DELETE 等
- **请求路径**：`/api/open/products`
- **请求参数**：查询参数和请求体参数
- **时间戳**：当前 Unix 时间戳（秒）
- **随机数**：随机生成的 16 位字符串

### 2. 参数排序

将所有请求参数（查询参数和请求体参数）按照字典序（lexicographical order）排序。

示例：
```json
{
  "productId": "123",
  "quantity": "2",
  "userId": "user_001"
}
```

排序后：
```
productId=123&quantity=2&userId=user_001
```

### 3. 构建签名字符串

将以下信息按顺序拼接：

```
{method}{path}{sorted_params}{timestamp}{nonce}
```

示例：
```
GET/api/open/productsproductId=123&quantity=2&userId=user_0011717459200a1b2c3d4e5f6g7h8
```

### 4. 生成签名

使用 HMAC-SHA256 算法，以上述签名字符串和 API Secret 生成签名，然后转换为十六进制小写字符串。

## Java 示例代码

```java
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;

public class ApiSignatureExample {
    
    private static final String HMAC_SHA256 = "HmacSHA256";
    
    /**
     * 生成 API 请求签名
     */
    public static String generateSignature(String method, String path, 
                                         Map<String, String> params, 
                                         long timestamp, String nonce, 
                                         String apiSecret) {
        try {
            // 1. 参数排序
            Map<String, String> sortedParams = new TreeMap<>(params);
            
            // 2. 构建参数字符串
            StringBuilder paramStr = new StringBuilder();
            for (Map.Entry<String, String> entry : sortedParams.entrySet()) {
                if (paramStr.length() > 0) {
                    paramStr.append("&");
                }
                paramStr.append(entry.getKey()).append("=").append(entry.getValue());
            }
            
            // 3. 构建签名字符串
            String signString = method + path + paramStr.toString() + timestamp + nonce;
            
            // 4. 生成 HMAC-SHA256 签名
            Mac mac = Mac.getInstance(HMAC_SHA256);
            SecretKeySpec secretKeySpec = new SecretKeySpec(
                apiSecret.getBytes(StandardCharsets.UTF_8), HMAC_SHA256);
            mac.init(secretKeySpec);
            
            byte[] hash = mac.doFinal(signString.getBytes(StandardCharsets.UTF_8));
            
            // 5. 转换为十六进制小写
            return bytesToHex(hash).toLowerCase();
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to generate signature", e);
        }
    }
    
    /**
     * 字节数组转十六进制
     */
    private static String bytesToHex(byte[] bytes) {
        StringBuilder hexString = new StringBuilder();
        for (byte b : bytes) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) {
                hexString.append('0');
            }
            hexString.append(hex);
        }
        return hexString.toString();
    }
    
    /**
     * 生成随机 nonce
     */
    public static String generateNonce() {
        return UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }
    
    /**
     * 使用示例
     */
    public static void main(String[] args) {
        // 配置信息
        String apiKey = "key_1234567890abcdef";
        String apiSecret = "secret_abcdef1234567890";
        String method = "GET";
        String path = "/api/open/products";
        
        // 请求参数
        Map<String, String> params = new HashMap<>();
        params.put("productId", "123");
        params.put("quantity", "2");
        params.put("userId", "user_001");
        
        // 生成时间戳和 nonce
        long timestamp = System.currentTimeMillis() / 1000;
        String nonce = generateNonce();
        
        // 生成签名
        String signature = generateSignature(method, path, params, timestamp, nonce, apiSecret);
        
        // 输出请求头
        System.out.println("X-API-Key: " + apiKey);
        System.out.println("X-API-Secret: " + apiSecret);
        System.out.println("X-Timestamp: " + timestamp);
        System.out.println("X-Nonce: " + nonce);
        System.out.println("X-Signature: " + signature);
    }
}
```

## cURL 示例

```bash
#!/bin/bash

# 配置信息
API_KEY="key_1234567890abcdef"
API_SECRET="secret_abcdef1234567890"
METHOD="GET"
PATH="/api/open/products"
BASE_URL="http://localhost:10081"

# 请求参数
PARAMS="productId=123&quantity=2"

# 生成时间戳和 nonce
TIMESTAMP=$(date +%s)
NONCE=$(uuidgen | tr -d '-' | tr '[:upper:]' '[:lower:]' | cut -c1-16)

# 构建签名字符串
SIGN_STRING="${METHOD}${PATH}${PARAMS}${TIMESTAMP}${NONCE}"

# 生成签名 (需要安装 openssl)
SIGNATURE=$(echo -n "$SIGN_STRING" | openssl dgst -sha256 -hmac "$API_SECRET" | cut -d' ' -f2)

# 发送请求
curl -X GET "${BASE_URL}${PATH}?${PARAMS}" \
  -H "X-API-Key: ${API_KEY}" \
  -H "X-API-Secret: ${API_SECRET}" \
  -H "X-Timestamp: ${TIMESTAMP}" \
  -H "X-Nonce: ${NONCE}" \
  -H "X-Signature: ${SIGNATURE}"
```

## Python 示例代码

```python
import hmac
import hashlib
import time
import uuid

def generate_signature(method, path, params, timestamp, nonce, api_secret):
    """生成 API 请求签名"""
    # 1. 参数排序
    sorted_params = dict(sorted(params.items()))
    
    # 2. 构建参数字符串
    param_str = '&'.join([f"{k}={v}" for k, v in sorted_params.items()])
    
    # 3. 构建签名字符串
    sign_string = f"{method}{path}{param_str}{timestamp}{nonce}"
    
    # 4. 生成 HMAC-SHA256 签名
    signature = hmac.new(
        api_secret.encode('utf-8'),
        sign_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest().lower()
    
    return signature

def generate_nonce():
    """生成随机 nonce"""
    return uuid.uuid4().hex[:16]

# 使用示例
api_key = "key_1234567890abcdef"
api_secret = "secret_abcdef1234567890"
method = "GET"
path = "/api/open/products"

# 请求参数
params = {
    "productId": "123",
    "quantity": "2",
    "userId": "user_001"
}

# 生成时间戳和 nonce
timestamp = int(time.time())
nonce = generate_nonce()

# 生成签名
signature = generate_signature(method, path, params, timestamp, nonce, api_secret)

print(f"X-API-Key: {api_key}")
print(f"X-API-Secret: {api_secret}")
print(f"X-Timestamp: {timestamp}")
print(f"X-Nonce: {nonce}")
print(f"X-Signature: {signature}")
```

## 安全注意事项

1. **保密 API Secret**：切勿在前端代码中硬编码 API Secret，应在后端生成签名
2. **时间戳验证**：服务器会验证时间戳，请求必须在 5 分钟内到达
3. **Nonce 唯一性**：每次请求应使用不同的 nonce，防止重放攻击
4. **HTTPS 传输**：生产环境必须使用 HTTPS 加密传输
5. **定期轮换密钥**：建议定期轮换 API Key 和 Secret

## 错误处理

如果签名验证失败，服务器将返回 `401 Unauthorized` 状态码，响应体示例：

```json
{
  "code": "INVALID_SIGNATURE",
  "message": "Request signature verification failed. Possible request tampering detected.",
  "path": "/api/open/products",
  "timestamp": "2026-06-03T20:30:00"
}
```

## 常见问题

### 1. 签名一直验证失败怎么办？

- 检查 API Secret 是否正确
- 确认参数排序是否正确（字典序）
- 确认时间戳是否为秒级（不是毫秒）
- 确认签名字符串构建是否正确

### 2. 时间戳验证失败怎么办？

- 检查服务器时间是否同步（使用 NTP）
- 确认时间戳是秒级 Unix 时间
- 确认请求在 5 分钟内到达服务器

### 3. 如何调试签名问题？

- 开启网关日志：`logging.level.com.metawebthree.gateway.auth=DEBUG`
- 使用相同参数在服务器端生成签名，对比客户端签名
- 检查请求头是否正确传递

## 联系方式

如有疑问，请联系技术支持：support@metawebthree.com
