package com.metawebthree.payment.application;

import java.util.Map;

public interface PaymentService {
    String createPayment(Object order);
    boolean verifyPaymentCallback(String paymentOrderNo, String signature, String data);
    String queryPaymentStatus(String paymentOrderNo);
    boolean refundPayment(String paymentOrderNo, String reason);
    Map<String, String> getWechatPayParams(Long orderId, Long userId);
    Map<String, String> getAlipayParams(Long orderId, Long userId);
    Map<String, String> getStripeParams(Long orderId, Long userId);
    boolean verifyPayment(String orderId, String transactionId, Long userId);
}