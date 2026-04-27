package com.metawebthree.payment.application;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class PaymentServiceImplTest {

    private PaymentServiceImpl paymentService;

    @BeforeEach
    void setUp() throws Exception {
        paymentService = new PaymentServiceImpl();
        setField("alipayAppId", "alipay-test-app");
        setField("wechatAppId", "wx-test-app");
    }

    @Test
    void getWechatPayParamsIncludesConfiguredAppId() {
        Map<String, String> params = paymentService.getWechatPayParams(1001L, 9L);

        assertEquals("wx-test-app", params.get("appId"));
        assertEquals("PREPAY_1001", params.get("prepayId"));
        assertEquals("Sign=WXPay", params.get("packageValue"));
    }

    @Test
    void verifyPaymentRejectsBlankTransactionId() {
        assertFalse(paymentService.verifyPayment("1001", "", 9L));
    }

    @Test
    void verifyPaymentAcceptsKnownPaymentArtifacts() {
        assertTrue(paymentService.verifyPayment("1001", "PREPAY_1001", 9L));
        assertTrue(paymentService.verifyPayment("1001", "ALIPAY_ORDER_STRING_1001", 9L));
        assertTrue(paymentService.verifyPayment("1001", "pi_mock_secret_1001", 9L));
    }

    private void setField(String name, String value) throws Exception {
        Field field = PaymentServiceImpl.class.getDeclaredField(name);
        field.setAccessible(true);
        field.set(paymentService, value);
    }
}
