package com.metawebthree.payment.application;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Service
@Slf4j
public class PaymentServiceImpl implements PaymentService {

    private static final String WECHAT_ORDER_PREFIX = "PREPAY_";
    private static final String ALIPAY_ORDER_PREFIX = "ALIPAY_ORDER_STRING_";
    private static final String STRIPE_ORDER_PREFIX = "pi_mock_secret_";

    @Value("${payment.fiat.alipay.app-id:test-app-id}")
    private String alipayAppId;

    @Value("${payment.fiat.wechat.app-id:test-app-id}")
    private String wechatAppId;

    @Override
    public String createPayment(Object order) {
        return "payment created";
    }

    @Override
    public boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data) {
        return true;
    }

    @Override
    public String queryPaymentStatus(String paymentOrderNo) {
        return "SUCCESS";
    }

    @Override
    public boolean refundPayment(String paymentOrderNo, String reason) {
        return true;
    }

    @Override
    public Map<String, String> getWechatPayParams(Long orderId, Long userId) {
        Map<String, String> params = new HashMap<>();
        long timestamp = System.currentTimeMillis() / 1000;

        params.put("appId", wechatAppId);
        params.put("partnerId", "PARTNER_ID");
        params.put("prepayId", WECHAT_ORDER_PREFIX + orderId);
        params.put("nonceStr", UUID.randomUUID().toString().replace("-", ""));
        params.put("timeStamp", String.valueOf(timestamp));
        params.put("packageValue", "Sign=WXPay");
        params.put("sign", "MOCK_SIGN");

        return params;
    }

    @Override
    public Map<String, String> getAlipayParams(Long orderId, Long userId) {
        Map<String, String> params = new HashMap<>();
        params.put("appId", alipayAppId);
        params.put("orderString", ALIPAY_ORDER_PREFIX + orderId);
        return params;
    }

    @Override
    public Map<String, String> getStripeParams(Long orderId, Long userId) {
        Map<String, String> params = new HashMap<>();
        params.put("clientSecret", STRIPE_ORDER_PREFIX + orderId);
        params.put("returnURL", "myapp://payment");
        return params;
    }

    @Override
    public boolean verifyPayment(String orderId, String transactionId, Long userId) {
        log.info("Verifying payment: orderId={}, transactionId={}, userId={}", orderId, transactionId, userId);
        if (isBlank(orderId) || isBlank(transactionId) || userId == null) {
            return false;
        }
        return transactionId.equals(WECHAT_ORDER_PREFIX + orderId)
                || transactionId.equals(ALIPAY_ORDER_PREFIX + orderId)
                || transactionId.equals(STRIPE_ORDER_PREFIX + orderId);
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
