package com.metawebthree.payment.application;

import com.alipay.api.AlipayApiException;
import com.alipay.api.DefaultAlipayClient;
import com.alipay.api.AlipayClient;
import com.alipay.api.request.AlipayTradeAppPayRequest;
import com.alipay.api.response.AlipayTradeAppPayResponse;
import com.alipay.api.AlipayConstants;
import com.alipay.api.internal.util.AlipaySignature;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import javax.servlet.http.HttpServletRequest;

@Service
@Slf4j
@RequiredArgsConstructor
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

    private final WebClient webClient;

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

    @Override
    public Map<String, Object> queryAlipayStatus(String outTradeNo, Long userId) {
        log.info("Querying Alipay status: outTradeNo={}, userId={}", outTradeNo, userId);

        Map<String, Object> result = new HashMap<>();
        result.put("outTradeNo", outTradeNo);
        result.put("tradeStatus", "TRADE_SUCCESS");
        result.put("totalAmount", "0.01");
        result.put("buyerLogonId", "user@example.com");
        return result;
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
                String orderServiceUrl = "http://order-service/order/paySuccess";

                Map<String, Object> request = new HashMap<>();
                request.put("orderId", orderId);
                request.put("payType", 2);

                webClient.post()
                    .uri(orderServiceUrl)
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();
                log.info("Order paySuccess called for orderId: {}", orderId);
            } catch (Exception e) {
                log.error("Failed to call order-service: {}", e.getMessage());
            }

            return "success";
        } catch (Exception e) {
            log.error("Alipay callback processing failed", e);
            return "failure";
        }
    }
}