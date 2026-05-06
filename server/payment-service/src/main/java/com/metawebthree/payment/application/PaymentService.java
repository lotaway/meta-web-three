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
    
    Map<String, Object> queryAlipayStatus(String outTradeNo, Long userId);
    
    /**
     * 处理支付宝异步回调
     * 
     * @param params 回调参数
     * @return "success" 表示处理成功，"failure" 表示处理失败
     */
    String handleAlipayCallback(Map<String, String> params);
}