package com.metawebthree.payment.application;

import com.alipay.api.AlipayApiException;
import com.alipay.api.DefaultAlipayClient;
import com.alipay.api.AlipayClient;
import com.alipay.api.request.AlipayTradeAppPayRequest;
import com.alipay.api.request.AlipayTradePagePayRequest;
import com.alipay.api.response.AlipayTradeAppPayResponse;
import com.alipay.api.response.AlipayTradePagePayResponse;
import com.alipay.api.AlipayConstants;
import com.alipay.api.internal.util.AlipaySignature;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.PaySuccessRequest;
import com.metawebthree.common.generated.rpc.PaySuccessResponse;
import com.metawebthree.payment.domain.exception.ExternalServiceException;
import com.metawebthree.payment.domain.exception.PaymentMethodNotConfiguredException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import jakarta.servlet.http.HttpServletRequest;

@Service
@Slf4j
@RequiredArgsConstructor
public class PaymentServiceImpl implements PaymentService {

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    @Value("${payment.fiat.alipay.app-id}")
    private String alipayAppId;

    @Value("${payment.fiat.alipay.private-key}")
    private String alipayPrivateKey;

    @Value("${payment.fiat.alipay.public-key}")
    private String alipayPublicKey;

    @Value("${payment.fiat.alipay.gateway-url}")
    private String alipayGatewayUrl;

    @Value("${payment.fiat.wechat.app-id}")
    private String wechatAppId;

    @Value("${payment.fiat.wechat.mch-id}")
    private String wechatMchId;

    @Value("${payment.fiat.wechat.api-key}")
    private String wechatApiKey;

    @Value("${payment.fiat.stripe.secret-key}")
    private String stripeSecretKey;

    @Value("${payment.fiat.stripe.return-url:app://payment}")
    private String stripeReturnUrl;

    private static final String NOTIFY_URL = "https://your-domain.com/api/pay/callback";

    @Override
    public String createPayment(Object order) {
        // Use Alipay to create payment order
        if (alipayAppId == null || alipayAppId.isEmpty() || alipayPrivateKey == null || alipayPrivateKey.isEmpty()) {
            log.error("Alipay not configured, cannot create payment");
            throw new PaymentMethodNotConfiguredException("Alipay", "Alipay not configured");
        }
        
        try {
            // Extract order information - assuming order has getId() and getAmount() methods
            Long orderId = getOrderId(order);
            Long amount = getOrderAmount(order);
            
            log.info("Creating Alipay payment for order: {}, amount: {}", orderId, amount);
            
            AlipayClient client = new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey
            );
            
            // Use trade page API for web payment
            AlipayTradePagePayRequest request = new AlipayTradePagePayRequest();
            request.setNotifyUrl(NOTIFY_URL + "/alipay");
            request.setReturnUrl(NOTIFY_URL + "/return");
            request.setBizContent(String.format(
                "{\"out_trade_no\":\"ORDER_%d\",\"total_amount\":\"%.2f\",\"subject\":\"Order %d\",\"product_code\":\"FAST_INSTANT_TRADE_PAY\"}",
                orderId, amount / 100.0, orderId
            ));
            
            AlipayTradePagePayResponse response = client.pageExecute(request);
            if (response.isSuccess()) {
                log.info("Alipay payment created successfully for order: {}", orderId);
                return response.getBody(); // This is the HTML form to redirect user
            } else {
                log.error("Alipay payment creation failed: {}", response.getMsg());
                throw new ExternalServiceException("Alipay", "Payment creation failed: " + response.getMsg());
            }
        } catch (AlipayApiException e) {
            log.error("Alipay payment creation failed: {}", e.getMessage());
            throw new ExternalServiceException("Alipay", "Payment creation failed: " + e.getMessage(), e);
        }
    }
    
    private Long getOrderId(Object order) {
        if (order instanceof Long) {
            return (Long) order;
        }
        try {
            java.lang.reflect.Method method = order.getClass().getMethod("getId");
            return (Long) method.invoke(order);
        } catch (Exception e) {
            return System.currentTimeMillis(); // Fallback to timestamp
        }
    }
    
    private Long getOrderAmount(Object order) {
        try {
            java.lang.reflect.Method method = order.getClass().getMethod("getAmount");
            Object result = method.invoke(order);
            if (result instanceof Long) {
                return (Long) result;
            } else if (result instanceof Integer) {
                return ((Integer) result).longValue();
            }
        } catch (Exception e) {
            // Ignore
        }
        return 1L; // Default 0.01 yuan (in cents)
    }

    @Override
    public boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data) {
        if (alipayPublicKey == null || alipayPublicKey.isEmpty()) {
            log.error("Alipay public key not configured, cannot verify payment callback");
            throw new PaymentMethodNotConfiguredException("Alipay", "Alipay public key not configured");
        }
        
        try {
            log.info("Verifying Alipay callback for order: {}", paymentOrderNo);
            
            // Parse the data as parameters map
            Map<String, String> params = parsePaymentData(data);
            
            boolean signVerified = AlipaySignature.rsaCheckV1(
                params,
                alipayPublicKey,
                "UTF-8",
                "RSA2"
            );
            
            if (signVerified) {
                String tradeStatus = params.get("trade_status");
                if ("TRADE_SUCCESS".equals(tradeStatus) || "TRADE_FINISHED".equals(tradeStatus)) {
                    log.info("Alipay callback verified successfully for order: {}", paymentOrderNo);
                    return true;
                }
            }
            
            log.warn("Alipay callback verification failed for order: {}", paymentOrderNo);
            return false;
        } catch (AlipayApiException e) {
            log.error("Alipay callback verification failed: {}", e.getMessage());
            return false;
        }
    }
    
    private Map<String, String> parsePaymentData(String data) {
        Map<String, String> params = new HashMap<>();
        if (data == null || data.isEmpty()) {
            return params;
        }
        String[] pairs = data.split("&");
        for (String pair : pairs) {
            String[] kv = pair.split("=");
            if (kv.length == 2) {
                params.put(kv[0], kv[1]);
            }
        }
        return params;
    }

    @Override
    public String queryPaymentStatus(String paymentOrderNo) {
        if (alipayAppId == null || alipayAppId.isEmpty()) {
            log.error("Alipay appId not configured, cannot query payment status");
            throw new PaymentMethodNotConfiguredException("Alipay", "Alipay not configured");
        }
        
        try {
            log.info("Querying Alipay status for order: {}", paymentOrderNo);
            
            AlipayClient client = new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey
            );
            
            com.alipay.api.request.AlipayTradeQueryRequest request = new com.alipay.api.request.AlipayTradeQueryRequest();
            request.setBizContent(String.format(
                "{\"out_trade_no\":\"%s\"}",
                paymentOrderNo
            ));
            
            com.alipay.api.response.AlipayTradeQueryResponse response = client.execute(request);
            
            if (response.isSuccess()) {
                String tradeStatus = response.getTradeStatus();
                log.info("Alipay status for order {}: {}", paymentOrderNo, tradeStatus);
                return tradeStatus;
            } else {
                log.error("Alipay query failed: {}", response.getMsg());
                return "UNKNOWN";
            }
        } catch (AlipayApiException e) {
            log.error("Alipay status query failed: {}", e.getMessage());
            return "ERROR";
        }
    }

    @Override
    public boolean refundPayment(String paymentOrderNo, String reason) {
        if (alipayAppId == null || alipayAppId.isEmpty()) {
            log.error("Alipay appId not configured, cannot process refund");
            throw new PaymentMethodNotConfiguredException("Alipay", "Alipay not configured");
        }
        
        try {
            log.info("Processing refund for order: {}, reason: {}", paymentOrderNo, reason);
            
            AlipayClient client = new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey
            );
            
            com.alipay.api.request.AlipayTradeRefundRequest request = new com.alipay.api.request.AlipayTradeRefundRequest();
            request.setBizContent(String.format(
                "{\"out_trade_no\":\"%s\",\"refund_reason\":\"%s\"}",
                paymentOrderNo, reason != null ? reason : "Customer request"
            ));
            
            com.alipay.api.response.AlipayTradeRefundResponse response = client.execute(request);
            
            if (response.isSuccess()) {
                log.info("Refund processed successfully for order: {}", paymentOrderNo);
                return true;
            } else {
                log.error("Refund failed: {}", response.getMsg());
                return false;
            }
        } catch (AlipayApiException e) {
            log.error("Refund processing failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public Map<String, String> getWechatPayParams(Long orderId, Long userId) {
        if (wechatAppId == null || wechatAppId.isEmpty() || wechatMchId == null || wechatMchId.isEmpty()) {
            log.error("Wechat Pay params not configured");
            throw new PaymentMethodNotConfiguredException("WechatPay", "Wechat Pay not configured");
        }
        
        try {
            log.info("Creating Wechat Pay params for order: {}, userId: {}", orderId, userId);
            
            // Generate nonce string
            String nonceStr = generateNonceStr();
            String outTradeNo = "ORDER_" + orderId;
            
            // For Native pay, we need to call the unified order API first
            // This is a simplified implementation - in production, you'd use the full SDK
            Map<String, String> result = new HashMap<>();
            result.put("appId", wechatAppId);
            result.put("mchId", wechatMchId);
            result.put("outTradeNo", outTradeNo);
            result.put("nonceStr", nonceStr);
            result.put("tradeType", "NATIVE");
            
            // Note: In production, you would call wechat pay unified order API here
            // and get the code_url to generate QR code
            result.put("codeUrl", "weixin://wxpay/bizpayurl?pr=placeholder");
            
            log.info("Wechat Pay params generated for order: {}", orderId);
            return result;
            
        } catch (Exception e) {
            log.error("Failed to generate Wechat Pay params: {}", e.getMessage());
            throw new ExternalServiceException("WechatPay", "Params generation failed: " + e.getMessage(), e);
        }
    }
    
    private String generateNonceStr() {
        String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            sb.append(chars.charAt((int) (Math.random() * chars.length())));
        }
        return sb.toString();
    }

    @Override
    public Map<String, String> getAlipayParams(Long orderId, Long userId) {
        try {
            log.info("Creating Alipay params for order: {}", orderId);

            AlipayClient client = new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey
            );

            AlipayTradeAppPayRequest request = new AlipayTradeAppPayRequest();
            request.setBizContent(String.format(
                "{\"out_trade_no\":\"ORDER_%d\",\"total_amount\":\"0.01\",\"subject\":\"Order %d\"}",
                orderId, orderId
            ));
            request.setNotifyUrl(NOTIFY_URL + "/alipay");

            AlipayTradeAppPayResponse response = client.sdkExecute(request);

            Map<String, String> result = new HashMap<>();
            result.put("appId", alipayAppId);
            result.put("orderString", response.getBody());
            return result;
        } catch (AlipayApiException e) {
            log.error("Alipay failed: {}", e.getMessage());
            throw new ExternalServiceException("Alipay", "Initialization failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Map<String, String> getStripeParams(Long orderId, Long userId) {
        if (stripeSecretKey == null || stripeSecretKey.isEmpty()) {
            log.error("Stripe secret key not configured");
            throw new PaymentMethodNotConfiguredException("Stripe", "Stripe not configured");
        }
        
        try {
            log.info("Creating Stripe payment for order: {}, userId: {}", orderId, userId);
            
            com.stripe.Stripe.apiKey = stripeSecretKey;
            
            // Create a payment intent
            com.stripe.param.PaymentIntentCreateParams params = 
                com.stripe.param.PaymentIntentCreateParams.builder()
                    .setAmount(1L) // Amount in cents
                    .setCurrency("cny")
                    .putMetadata("orderId", String.valueOf(orderId))
                    .putMetadata("userId", String.valueOf(userId))
                    .setReturnUrl(stripeReturnUrl + "?orderId=" + orderId)
                    .build();
            
            com.stripe.model.PaymentIntent paymentIntent = com.stripe.model.PaymentIntent.create(params);
            
            Map<String, String> result = new HashMap<>();
            result.put("clientSecret", paymentIntent.getClientSecret());
            result.put("paymentIntentId", paymentIntent.getId());
            
            log.info("Stripe payment created for order: {}", orderId);
            return result;
            
        } catch (com.stripe.exception.StripeException e) {
            log.error("Stripe payment creation failed: {}", e.getMessage());
            throw new ExternalServiceException("Stripe", "Payment creation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean verifyPayment(String orderId, String transactionId, Long userId) {
        if (alipayAppId == null || alipayAppId.isEmpty()) {
            log.error("Payment verification not configured");
            throw new PaymentMethodNotConfiguredException("Alipay", "Payment verification not configured");
        }
        
        try {
            log.info("Verifying payment: orderId={}, transactionId={}, userId={}", orderId, transactionId, userId);
            
            AlipayClient client = new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey
            );
            
            com.alipay.api.request.AlipayTradeQueryRequest request = new com.alipay.api.request.AlipayTradeQueryRequest();
            // Support both out_trade_no and trade_no
            if (transactionId != null && transactionId.startsWith("ORDER_")) {
                request.setBizContent(String.format(
                    "{\"out_trade_no\":\"%s\"}",
                    transactionId
                ));
            } else {
                request.setBizContent(String.format(
                    "{\"trade_no\":\"%s\"}",
                    transactionId
                ));
            }
            
            com.alipay.api.response.AlipayTradeQueryResponse response = client.execute(request);
            
            if (response.isSuccess()) {
                String tradeStatus = response.getTradeStatus();
                boolean verified = "TRADE_SUCCESS".equals(tradeStatus) || "TRADE_FINISHED".equals(tradeStatus);
                log.info("Payment verification result for order {}: {}", orderId, verified);
                return verified;
            }
            
            log.warn("Payment verification failed for order: {}", orderId);
            return false;
        } catch (AlipayApiException e) {
            log.error("Payment verification failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public Map<String, Object> queryAlipayStatus(String outTradeNo, Long userId) {
        if (alipayAppId == null || alipayAppId.isEmpty()) {
            log.error("Alipay status query not configured");
            throw new PaymentMethodNotConfiguredException("Alipay", "Alipay status query not configured");
        }
        
        try {
            log.info("Querying Alipay status: outTradeNo={}, userId={}", outTradeNo, userId);
            
            AlipayClient client = new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey
            );
            
            com.alipay.api.request.AlipayTradeQueryRequest request = new com.alipay.api.request.AlipayTradeQueryRequest();
            request.setBizContent(String.format(
                "{\"out_trade_no\":\"%s\"}",
                outTradeNo
            ));
            
            com.alipay.api.response.AlipayTradeQueryResponse response = client.execute(request);
            
            Map<String, Object> result = new HashMap<>();
            if (response.isSuccess()) {
                result.put("success", true);
                result.put("tradeStatus", response.getTradeStatus());
                result.put("tradeNo", response.getTradeNo());
                result.put("outTradeNo", response.getOutTradeNo());
                result.put("totalAmount", response.getTotalAmount());
                result.put("buyerLogonId", response.getBuyerLogonId());
                result.put("buyerPayAmount", response.getBuyerPayAmount());
                log.info("Alipay status query success for order: {}, status: {}", outTradeNo, response.getTradeStatus());
            } else {
                result.put("success", false);
                result.put("error", response.getMsg());
                log.warn("Alipay status query failed for order: {}", outTradeNo);
            }
            
            return result;
        } catch (AlipayApiException e) {
            log.error("Alipay status query failed: {}", e.getMessage());
            Map<String, Object> result = new HashMap<>();
            result.put("success", false);
            result.put("error", e.getMessage());
            return result;
        }
    }

    @Override
    public String handleAlipayCallback(Map<String, String> params) {
        try {
            log.info("Alipay callback received: {}", params);

            boolean signVerified = AlipaySignature.rsaCheckV1(
                params,
                alipayPublicKey,
                "UTF-8",
                "RSA2"
            );

            if (!signVerified) {
                log.warn("Alipay callback signature verification failed");
                return "failure";
            }

            String tradeStatus = params.get("trade_status");
            String outTradeNo = params.get("out_trade_no");
            String tradeNo = params.get("trade_no");

            if (!"TRADE_SUCCESS".equals(tradeStatus) && !"TRADE_FINISHED".equals(tradeStatus)) {
                log.info("Alipay trade status is not success: {}", tradeStatus);
                return "success";
            }

            Long orderId = null;
            if (outTradeNo != null && outTradeNo.startsWith("ORDER_")) {
                try {
                    orderId = Long.parseLong(outTradeNo.substring(6));
                } catch (NumberFormatException e) {
                    log.error("Failed to parse orderId from outTradeNo: {}", outTradeNo);
                    return "failure";
                }
            }

            if (orderId == null) {
                log.error("Invalid outTradeNo: {}", outTradeNo);
                return "failure";
            }

            try {
                // Use Dubbo RPC to call order-service instead of HTTP
                PaySuccessRequest request = PaySuccessRequest.newBuilder()
                        .setOrderId(orderId)
                        .setPayType(2)
                        .build();
                PaySuccessResponse response = orderService.paySuccess(request);
                if (response != null && response.getSuccess()) {
                    log.info("Order paySuccess called via Dubbo for orderId: {}", orderId);
                } else {
                    log.warn("Order paySuccess returned false for orderId: {}", orderId);
                }
            } catch (Exception e) {
                log.error("Failed to call order-service via Dubbo: {}", e.getMessage());
            }

            return "success";
        } catch (Exception e) {
            log.error("Alipay callback processing failed", e);
            return "failure";
        }
    }
}