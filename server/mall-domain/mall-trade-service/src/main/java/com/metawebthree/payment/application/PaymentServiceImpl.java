package com.metawebthree.payment.application;

import com.alipay.api.AlipayApiException;
import com.alipay.api.DefaultAlipayClient;
import com.alipay.api.AlipayClient;
import com.alipay.api.request.AlipayTradeAppPayRequest;
import com.alipay.api.request.AlipayTradePagePayRequest;
import com.alipay.api.response.AlipayTradeAppPayResponse;
import com.alipay.api.response.AlipayTradePagePayResponse;
import com.alipay.api.internal.util.AlipaySignature;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.PaySuccessRequest;
import com.metawebthree.common.generated.rpc.PaySuccessResponse;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import com.metawebthree.payment.infrastructure.idempotency.PaymentNotificationIdempotencyStore;
import com.metawebthree.payment.domain.exception.ExternalServiceException;
import com.metawebthree.payment.domain.exception.PaymentMethodNotConfiguredException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;

/**
 * Fiat payment integration.
 *
 * <p>Design rules enforced here:</p>
 * <ul>
 *   <li>Amounts always come from the real order ({@link OrderDO#getOrderAmount()}),
 *       never from a hard-coded placeholder.</li>
 *   <li>Credentials and callback URLs are injected from the environment. A method
 *       that is not fully configured is explicitly disabled by throwing
 *       {@link PaymentMethodNotConfiguredException} instead of returning fabricated
 *       payment parameters.</li>
 * </ul>
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class PaymentServiceImpl implements PaymentService {

    private static final String ALIPAY_PRODUCT_CODE_PAGE = "FAST_INSTANT_TRADE_PAY";
    private static final String ALIPAY_PRODUCT_CODE_APP = "QUICK_MSECURITY_PAY";

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    private final PaymentNotificationIdempotencyStore idempotencyStore;
    private final OrderMapper orderMapper;

    @Value("${payment.fiat.alipay.app-id:}")
    private String alipayAppId;

    @Value("${payment.fiat.alipay.private-key:}")
    private String alipayPrivateKey;

    @Value("${payment.fiat.alipay.public-key:}")
    private String alipayPublicKey;

    @Value("${payment.fiat.alipay.gateway-url:https://openapi.alipay.com/gateway.do}")
    private String alipayGatewayUrl;

    /** Exact Alipay async notify URL (no suffix is appended). */
    @Value("${payment.fiat.alipay.notify-url:}")
    private String alipayNotifyUrl;

    /** Optional synchronous return URL. */
    @Value("${payment.fiat.alipay.return-url:}")
    private String alipayReturnUrl;

    /** WeChat Pay is disabled until a real unified-order integration is provided. */
    @Value("${payment.fiat.wechat.enabled:false}")
    private boolean wechatEnabled;

    @Value("${payment.fiat.wechat.app-id:}")
    private String wechatAppId;

    @Value("${payment.fiat.wechat.mch-id:}")
    private String wechatMchId;

    @Value("${payment.fiat.wechat.api-key:}")
    private String wechatApiKey;

    @Value("${payment.fiat.stripe.secret-key:}")
    private String stripeSecretKey;

    @Value("${payment.fiat.stripe.return-url:app://payment}")
    private String stripeReturnUrl;

    @Value("${payment.fiat.stripe.currency:cny}")
    private String stripeCurrency;

    @Override
    public String createPayment(Object order) {
        requireAlipayConfigured();
        requireAlipayNotifyUrl();

        Long orderId = resolveOrderId(order);
        BigDecimal amount = resolveAmountYuan(order);
        log.info("Creating Alipay page payment for order: {}, amount: {}", orderId, amount);

        try {
            AlipayTradePagePayRequest request = new AlipayTradePagePayRequest();
            request.setNotifyUrl(alipayNotifyUrl);
            if (StringUtils.hasText(alipayReturnUrl)) {
                request.setReturnUrl(alipayReturnUrl);
            }
            request.setBizContent(buildAlipayOrderBizContent(
                    "ORDER_" + orderId, amount, orderId, ALIPAY_PRODUCT_CODE_PAGE));

            AlipayTradePagePayResponse response = alipayClient().pageExecute(request);
            if (response.isSuccess()) {
                log.info("Alipay page payment created successfully for order: {}", orderId);
                return response.getBody();
            }
            log.error("Alipay page payment creation failed: {}", response.getMsg());
            throw new ExternalServiceException("Alipay", "Payment creation failed: " + response.getMsg());
        } catch (AlipayApiException e) {
            log.error("Alipay page payment creation failed: {}", e.getMessage());
            throw new ExternalServiceException("Alipay", "Payment creation failed: " + e.getMessage(), e);
        }
    }

    /**
     * Builds the Alipay biz_content using the real order amount. Package-private so
     * the amount formatting can be unit tested without a live Alipay call.
     */
    String buildAlipayOrderBizContent(String outTradeNo, BigDecimal amountYuan, Long orderId, String productCode) {
        return String.format(
                "{\"out_trade_no\":\"%s\",\"total_amount\":\"%s\",\"subject\":\"%s\",\"product_code\":\"%s\"}",
                outTradeNo,
                amountYuan.setScale(2, RoundingMode.HALF_UP).toPlainString(),
                "Order " + orderId,
                productCode);
    }

    @Override
    public boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data) {
        requireAlipayPublicKey();

        try {
            log.info("Verifying Alipay callback for order: {}", paymentOrderNo);

            Map<String, String> params = parsePaymentData(data);

            boolean signVerified = AlipaySignature.rsaCheckV1(params, alipayPublicKey, "UTF-8", "RSA2");

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
        requireAlipayConfigured();
        try {
            log.info("Querying Alipay status for order: {}", paymentOrderNo);

            com.alipay.api.request.AlipayTradeQueryRequest request = new com.alipay.api.request.AlipayTradeQueryRequest();
            request.setBizContent(String.format("{\"out_trade_no\":\"%s\"}", paymentOrderNo));

            com.alipay.api.response.AlipayTradeQueryResponse response = alipayClient().execute(request);
            if (response.isSuccess()) {
                String tradeStatus = response.getTradeStatus();
                log.info("Alipay status for order {}: {}", paymentOrderNo, tradeStatus);
                return tradeStatus;
            }
            log.error("Alipay query failed: {}", response.getMsg());
            return "UNKNOWN";
        } catch (AlipayApiException e) {
            log.error("Alipay status query failed: {}", e.getMessage());
            return "ERROR";
        }
    }

    @Override
    public boolean refundPayment(String paymentOrderNo, String reason) {
        requireAlipayConfigured();
        try {
            log.info("Processing refund for order: {}, reason: {}", paymentOrderNo, reason);

            com.alipay.api.request.AlipayTradeRefundRequest request = new com.alipay.api.request.AlipayTradeRefundRequest();
            request.setBizContent(String.format(
                    "{\"out_trade_no\":\"%s\",\"refund_reason\":\"%s\"}",
                    paymentOrderNo, reason != null ? reason : "Customer request"));

            com.alipay.api.response.AlipayTradeRefundResponse response = alipayClient().execute(request);
            if (response.isSuccess()) {
                log.info("Refund processed successfully for order: {}", paymentOrderNo);
                return true;
            }
            log.error("Refund failed: {}", response.getMsg());
            return false;
        } catch (AlipayApiException e) {
            log.error("Refund processing failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public Map<String, String> getWechatPayParams(Long orderId, Long userId) {
        // WeChat Pay is intentionally disabled: there is no working unified-order
        // integration, so returning placeholder parameters would fake a payment.
        if (!wechatEnabled) {
            throw new PaymentMethodNotConfiguredException("WechatPay",
                    "Wechat Pay is disabled (payment.fiat.wechat.enabled=false)");
        }
        if (!StringUtils.hasText(wechatAppId)
                || !StringUtils.hasText(wechatMchId)
                || !StringUtils.hasText(wechatApiKey)) {
            throw new PaymentMethodNotConfiguredException("WechatPay",
                    "Wechat Pay is enabled but not fully configured");
        }
        throw new PaymentMethodNotConfiguredException("WechatPay",
                "Wechat Pay unified-order integration is not implemented in this build; "
                        + "no payment parameters are produced");
    }

    @Override
    public Map<String, String> getAlipayParams(Long orderId, Long userId) {
        requireAlipayConfigured();
        requireAlipayNotifyUrl();

        Long resolvedOrderId = requireOrderId(orderId);
        BigDecimal amount = loadOrderAmount(resolvedOrderId);
        log.info("Creating Alipay app params for order: {}, amount: {}", resolvedOrderId, amount);

        try {
            AlipayTradeAppPayRequest request = new AlipayTradeAppPayRequest();
            request.setNotifyUrl(alipayNotifyUrl);
            request.setBizContent(buildAlipayOrderBizContent(
                    "ORDER_" + resolvedOrderId, amount, resolvedOrderId, ALIPAY_PRODUCT_CODE_APP));

            AlipayTradeAppPayResponse response = alipayClient().sdkExecute(request);

            Map<String, String> result = new HashMap<>();
            result.put("orderString", response.getBody());
            return result;
        } catch (AlipayApiException e) {
            log.error("Creating Alipay app params failed: {}", e.getMessage());
            throw new ExternalServiceException("Alipay", "Params generation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Map<String, String> getStripeParams(Long orderId, Long userId) {
        if (!StringUtils.hasText(stripeSecretKey)) {
            throw new PaymentMethodNotConfiguredException("Stripe",
                    "Stripe secret key is not configured");
        }

        Long resolvedOrderId = requireOrderId(orderId);
        long amountMinorUnits = toMinorUnits(loadOrderAmount(resolvedOrderId));
        log.info("Creating Stripe payment intent for order: {}, amount(minor): {}, currency: {}",
                resolvedOrderId, amountMinorUnits, stripeCurrency);

        try {
            com.stripe.Stripe.apiKey = stripeSecretKey;

            com.stripe.param.PaymentIntentCreateParams params =
                    com.stripe.param.PaymentIntentCreateParams.builder()
                            .setAmount(amountMinorUnits)
                            .setCurrency(stripeCurrency)
                            .putMetadata("orderId", String.valueOf(resolvedOrderId))
                            .putMetadata("userId", String.valueOf(userId))
                            .setReturnUrl(stripeReturnUrl + "?orderId=" + resolvedOrderId)
                            .build();

            com.stripe.model.PaymentIntent paymentIntent = com.stripe.model.PaymentIntent.create(params);

            Map<String, String> result = new HashMap<>();
            result.put("clientSecret", paymentIntent.getClientSecret());
            result.put("paymentIntentId", paymentIntent.getId());
            return result;
        } catch (com.stripe.exception.StripeException e) {
            log.error("Stripe payment creation failed: {}", e.getMessage());
            throw new ExternalServiceException("Stripe", "Payment creation failed: " + e.getMessage(), e);
        }
    }

    /** Converts a yuan amount to the currency's minor unit (fen/cents). */
    long toMinorUnits(BigDecimal amountYuan) {
        if (amountYuan == null) {
            throw new IllegalArgumentException("Amount must not be null");
        }
        long minor = amountYuan.multiply(BigDecimal.valueOf(100))
                .setScale(0, RoundingMode.HALF_UP)
                .longValueExact();
        if (minor <= 0) {
            throw new IllegalArgumentException("Amount must be positive: " + amountYuan);
        }
        return minor;
    }

    @Override
    public boolean verifyPayment(String orderId, String transactionId, Long userId) {
        requireAlipayConfigured();
        try {
            log.info("Verifying payment: orderId={}, transactionId={}, userId={}", orderId, transactionId, userId);

            com.alipay.api.request.AlipayTradeQueryRequest request = new com.alipay.api.request.AlipayTradeQueryRequest();
            if (transactionId != null && transactionId.startsWith("ORDER_")) {
                request.setBizContent(String.format("{\"out_trade_no\":\"%s\"}", transactionId));
            } else {
                request.setBizContent(String.format("{\"trade_no\":\"%s\"}", transactionId));
            }

            com.alipay.api.response.AlipayTradeQueryResponse response = alipayClient().execute(request);
            if (response.isSuccess()) {
                String tradeStatus = response.getTradeStatus();
                boolean verified = "TRADE_SUCCESS".equals(tradeStatus) || "TRADE_FINISHED".equals(tradeStatus);
                log.info("Payment verification result for order {}: {}", orderId, verified);
                if (verified) {
                    try {
                        confirmOrderPaid(Long.parseLong(orderId), response.getTradeNo(), "ALIPAY", response.getTotalAmount());
                    } catch (NumberFormatException e) {
                        log.warn("Invalid orderId in payment verification: {}", orderId);
                    }
                }
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
        requireAlipayConfigured();
        try {
            log.info("Querying Alipay status: outTradeNo={}, userId={}", outTradeNo, userId);

            com.alipay.api.request.AlipayTradeQueryRequest request = new com.alipay.api.request.AlipayTradeQueryRequest();
            request.setBizContent(String.format("{\"out_trade_no\":\"%s\"}", outTradeNo));

            com.alipay.api.response.AlipayTradeQueryResponse response = alipayClient().execute(request);

            Map<String, Object> result = new HashMap<>();
            if (response.isSuccess()) {
                result.put("success", true);
                result.put("tradeStatus", response.getTradeStatus());
                result.put("tradeNo", response.getTradeNo());
                result.put("outTradeNo", response.getOutTradeNo());
                result.put("totalAmount", response.getTotalAmount());
                result.put("buyerPayAmount", response.getBuyerPayAmount());
                log.info("Alipay status query success for order: {}", outTradeNo);
                String queriedTradeStatus = response.getTradeStatus();
                if ("TRADE_SUCCESS".equals(queriedTradeStatus) || "TRADE_FINISHED".equals(queriedTradeStatus)) {
                    Long orderIdFromQuery = parseOrderIdFromOutTradeNo(outTradeNo);
                    if (orderIdFromQuery != null) {
                        confirmOrderPaid(orderIdFromQuery, response.getTradeNo(), "ALIPAY", response.getTotalAmount());
                    }
                }
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
        requireAlipayPublicKey();
        try {
            log.info("Alipay callback received: {}", params);

            boolean signVerified = AlipaySignature.rsaCheckV1(params, alipayPublicKey, "UTF-8", "RSA2");
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

            if (tradeNo == null || tradeNo.isBlank()) {
                log.error("Missing trade_no in alipay callback for outTradeNo: {}", outTradeNo);
                return "failure";
            }

            String callbackAppId = params.get("app_id");
            if (StringUtils.hasText(alipayAppId) && !alipayAppId.equals(callbackAppId)) {
                log.error("Alipay callback app_id mismatch: expected={}, actual={}", alipayAppId, callbackAppId);
                return "failure";
            }

            if (!idempotencyStore.tryRecord("ALIPAY", tradeNo, outTradeNo)) {
                log.info("Duplicate alipay callback ignored: tradeNo={}, outTradeNo={}", tradeNo, outTradeNo);
                return "success";
            }

            confirmOrderPaid(orderId, tradeNo, "ALIPAY", params.get("total_amount"));
            return "success";
        } catch (AlipayApiException e) {
            log.error("Alipay callback verification failed: {}", e.getMessage());
            return "failure";
        }
    }

    // ==================== Helpers ====================

    /** Alipay client factory. Overridable for tests. */
    protected AlipayClient alipayClient() {
        return new DefaultAlipayClient(
                alipayGatewayUrl, alipayAppId, alipayPrivateKey, "JSON", "UTF-8", alipayPublicKey);
    }

    private void requireAlipayConfigured() {
        if (!StringUtils.hasText(alipayAppId)
                || !StringUtils.hasText(alipayPrivateKey)
                || !StringUtils.hasText(alipayPublicKey)) {
            throw new PaymentMethodNotConfiguredException("Alipay",
                    "Alipay is not configured (app-id/private-key/public-key missing)");
        }
    }

    private void requireAlipayPublicKey() {
        if (!StringUtils.hasText(alipayPublicKey)) {
            throw new PaymentMethodNotConfiguredException("Alipay",
                    "Alipay public key is not configured");
        }
    }

    private void requireAlipayNotifyUrl() {
        if (!StringUtils.hasText(alipayNotifyUrl)) {
            throw new PaymentMethodNotConfiguredException("Alipay",
                    "Alipay notify URL is not configured");
        }
    }

    private Long requireOrderId(Long orderId) {
        if (orderId == null) {
            throw new IllegalArgumentException("orderId is required");
        }
        return orderId;
    }

    Long resolveOrderId(Object order) {
        if (order instanceof Number number) {
            return number.longValue();
        }
        Object id = invokeNoArg(order, "getId");
        if (id instanceof Number number) {
            return number.longValue();
        }
        Object altId = invokeNoArg(order, "getOrderId");
        if (altId instanceof Number number) {
            return number.longValue();
        }
        throw new IllegalArgumentException("Cannot resolve order id from: " + order);
    }

    BigDecimal resolveAmountYuan(Object order) {
        for (String accessor : new String[]{"getAmount", "getFiatAmount", "getOrderAmount", "getPayAmount"}) {
            Object value = invokeNoArg(order, accessor);
            if (value instanceof BigDecimal amount && amount.signum() > 0) {
                return amount;
            }
            if (value instanceof Number number && number.doubleValue() > 0) {
                return BigDecimal.valueOf(number.doubleValue());
            }
        }
        Object id = invokeNoArg(order, "getId");
        if (id instanceof Number number) {
            return loadOrderAmount(number.longValue());
        }
        throw new IllegalStateException("Cannot determine payment amount for order: " + order);
    }

    BigDecimal loadOrderAmount(Long orderId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null) {
            throw new IllegalStateException("Order not found: " + orderId);
        }
        BigDecimal amount = order.getOrderAmount();
        if (amount == null || amount.signum() <= 0) {
            throw new IllegalStateException("Order has no payable amount: " + orderId);
        }
        return amount;
    }

    private Object invokeNoArg(Object target, String methodName) {
        if (target == null) {
            return null;
        }
        try {
            Method method = target.getClass().getMethod(methodName);
            return method.invoke(target);
        } catch (Exception e) {
            return null;
        }
    }

    private void confirmOrderPaid(Long orderId, String tradeNo, String channel, String totalAmount) {
        try {
            PaySuccessRequest.Builder builder = PaySuccessRequest.newBuilder()
                    .setOrderId(orderId)
                    .setPayType(2);
            if (StringUtils.hasText(tradeNo)) {
                builder.setTransactionId(tradeNo);
            }
            if (StringUtils.hasText(channel)) {
                builder.setChannel(channel);
            }
            if (StringUtils.hasText(totalAmount)) {
                builder.setTotalAmount(totalAmount);
            }
            PaySuccessResponse response = orderService.paySuccess(builder.build());
            if (response != null && response.getSuccess()) {
                log.info("Order confirmed as paid via Dubbo for orderId: {}", orderId);
            } else {
                log.warn("Order paySuccess returned false for orderId: {}", orderId);
            }
        } catch (Exception e) {
            log.error("Failed to call order-service via Dubbo for orderId: {}: {}", orderId, e.getMessage(), e);
        }
    }

    private Long parseOrderIdFromOutTradeNo(String outTradeNo) {
        if (outTradeNo == null || !outTradeNo.startsWith("ORDER_")) {
            log.warn("Invalid outTradeNo for order confirmation: {}", outTradeNo);
            return null;
        }
        try {
            return Long.parseLong(outTradeNo.substring(6));
        } catch (NumberFormatException e) {
            log.warn("Failed to parse outTradeNo: {}", outTradeNo);
            return null;
        }
    }
}
