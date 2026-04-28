package com.metawebthree.payment.application;

import com.alipay.api.AlipayApiException;
import com.alipay.api.DefaultAlipayClient;
import com.alipay.api.AlipayClient;
import com.alipay.api.request.AlipayTradeAppPayRequest;
import com.alipay.api.response.AlipayTradeAppPayResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@Slf4j
public class PaymentServiceImpl implements PaymentService {

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
        log.info("Creating Wechat Pay params for order: {}", orderId);

        Map<String, String> result = new HashMap<>();
        result.put("appId", wechatAppId);
        result.put("partnerId", wechatMchId);
        result.put("prepayId", "PREPAY_" + orderId);
        result.put("nonceStr", String.valueOf(System.nanoTime()));
        result.put("timeStamp", String.valueOf(System.currentTimeMillis() / 1000));
        result.put("packageValue", "Sign=WXPay");
        result.put("sign", "use_api_key_sign");

        return result;
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
            throw new RuntimeException("支付宝支付初始化失败", e);
        }
    }

    @Override
    public Map<String, String> getStripeParams(Long orderId, Long userId) {
        log.info("Creating Stripe params for order: {}", orderId);

        Map<String, String> result = new HashMap<>();
        result.put("clientSecret", "pi_" + orderId + "_secret_" + stripeSecretKey.substring(0, 8));
        result.put("returnURL", stripeReturnUrl);
        return result;
    }

    @Override
    public boolean verifyPayment(String orderId, String transactionId, Long userId) {
        log.info("Verifying payment: orderId={}, transactionId={}, userId={}", orderId, transactionId, userId);
        return orderId != null && transactionId != null && userId != null;
    }
}