package com.metawebthree.payment.application;

import com.alipay.api.AlipayClient;
import com.alipay.api.request.AlipayTradeAppPayRequest;
import com.alipay.api.request.AlipayTradePagePayRequest;
import com.alipay.api.response.AlipayTradeAppPayResponse;
import com.alipay.api.response.AlipayTradePagePayResponse;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import com.metawebthree.payment.domain.exception.PaymentMethodNotConfiguredException;
import com.metawebthree.payment.infrastructure.idempotency.PaymentNotificationIdempotencyStore;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies that payment methods only produce real parameters or are explicitly
 * disabled, and that amounts come from the actual order.
 */
class PaymentMethodConfigurationTest {

    private final PaymentNotificationIdempotencyStore idempotencyStore =
            mock(PaymentNotificationIdempotencyStore.class);
    private final OrderMapper orderMapper = mock(OrderMapper.class);

    /** Test double that lets a mocked Alipay client be injected. */
    private static class TestPaymentService extends PaymentServiceImpl {
        private final AlipayClient client;

        TestPaymentService(PaymentNotificationIdempotencyStore store, OrderMapper mapper, AlipayClient client) {
            super(store, mapper);
            this.client = client;
        }

        @Override
        protected AlipayClient alipayClient() {
            return client;
        }
    }

    private static void setField(Object target, String name, Object value) {
        try {
            Field field = PaymentServiceImpl.class.getDeclaredField(name);
            field.setAccessible(true);
            field.set(target, value);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set field " + name, e);
        }
    }

    private TestPaymentService service(AlipayClient client) {
        return new TestPaymentService(idempotencyStore, orderMapper, client);
    }

    private TestPaymentService configuredAlipay(AlipayClient client) {
        TestPaymentService service = service(client);
        setField(service, "alipayAppId", "app-1");
        setField(service, "alipayPrivateKey", "private-key");
        setField(service, "alipayPublicKey", "public-key");
        setField(service, "alipayNotifyUrl", "https://api.example.com/pay/alipay/callback");
        return service;
    }

    private void givenOrder(long id, String amount) {
        when(orderMapper.selectById(id))
                .thenReturn(OrderDO.builder().id(id).orderAmount(new BigDecimal(amount)).build());
    }

    @Test
    void wechatPayIsExplicitlyDisabledRatherThanReturningPlaceholder() {
        TestPaymentService service = service(mock(AlipayClient.class));
        setField(service, "wechatEnabled", false);

        PaymentMethodNotConfiguredException ex = assertThrows(PaymentMethodNotConfiguredException.class,
                () -> service.getWechatPayParams(1L, 1L));

        assertTrue(ex.getMessage().toLowerCase().contains("disabled"), ex.getMessage());
    }

    @Test
    void wechatPayDoesNotFakeParamsEvenWhenEnabledButUnimplemented() {
        TestPaymentService service = service(mock(AlipayClient.class));
        setField(service, "wechatEnabled", true);
        setField(service, "wechatAppId", "wx-app");
        setField(service, "wechatMchId", "wx-mch");
        setField(service, "wechatApiKey", "wx-key");

        assertThrows(PaymentMethodNotConfiguredException.class,
                () -> service.getWechatPayParams(1L, 1L));
    }

    @Test
    void alipayParamsAreDisabledWhenNotConfigured() {
        TestPaymentService service = service(mock(AlipayClient.class));

        assertThrows(PaymentMethodNotConfiguredException.class,
                () -> service.getAlipayParams(1L, 1L));
    }

    @Test
    void alipayParamsRequireNotifyUrl() {
        TestPaymentService service = service(mock(AlipayClient.class));
        setField(service, "alipayAppId", "app-1");
        setField(service, "alipayPrivateKey", "private-key");
        setField(service, "alipayPublicKey", "public-key");

        assertThrows(PaymentMethodNotConfiguredException.class,
                () -> service.getAlipayParams(1L, 1L));
    }

    @Test
    void stripeParamsAreDisabledWhenNotConfigured() {
        TestPaymentService service = service(mock(AlipayClient.class));

        assertThrows(PaymentMethodNotConfiguredException.class,
                () -> service.getStripeParams(1L, 1L));
    }

    @Test
    void createPaymentIsDisabledWhenNotConfigured() {
        TestPaymentService service = service(mock(AlipayClient.class));

        assertThrows(PaymentMethodNotConfiguredException.class,
                () -> service.createPayment(new FakeOrder(7L, new BigDecimal("12.34"))));
    }

    @Test
    void alipayAppParamsUseRealOrderAmount() throws Exception {
        AlipayClient client = mock(AlipayClient.class);
        AlipayTradeAppPayResponse response = mock(AlipayTradeAppPayResponse.class);
        when(response.getBody()).thenReturn("orderString");
        when(client.sdkExecute(any(AlipayTradeAppPayRequest.class))).thenReturn(response);
        givenOrder(7L, "12.34");
        TestPaymentService service = configuredAlipay(client);

        Map<String, String> result = service.getAlipayParams(7L, 1L);

        assertEquals("orderString", result.get("orderString"));
        ArgumentCaptor<AlipayTradeAppPayRequest> captor = ArgumentCaptor.forClass(AlipayTradeAppPayRequest.class);
        org.mockito.Mockito.verify(client).sdkExecute(captor.capture());
        String bizContent = captor.getValue().getBizContent();
        assertTrue(bizContent.contains("\"total_amount\":\"12.34\""), bizContent);
        assertTrue(bizContent.contains("\"out_trade_no\":\"ORDER_7\""), bizContent);
    }

    @Test
    void createPaymentUsesRealOrderAmount() throws Exception {
        AlipayClient client = mock(AlipayClient.class);
        AlipayTradePagePayResponse response = mock(AlipayTradePagePayResponse.class);
        when(response.isSuccess()).thenReturn(true);
        when(response.getBody()).thenReturn("<html>pay</html>");
        when(client.pageExecute(any())).thenReturn(response);

        TestPaymentService service = configuredAlipay(client);

        String body = service.createPayment(new FakeOrder(7L, new BigDecimal("12.34")));

        assertEquals("<html>pay</html>", body);
        ArgumentCaptor<AlipayTradePagePayRequest> captor =
                ArgumentCaptor.forClass(AlipayTradePagePayRequest.class);
        org.mockito.Mockito.verify(client).pageExecute(captor.capture());
        String bizContent = captor.getValue().getBizContent();
        assertTrue(bizContent.contains("\"total_amount\":\"12.34\""), bizContent);
        assertTrue(bizContent.contains("\"out_trade_no\":\"ORDER_7\""), bizContent);
    }

    @Test
    void alipayAmountMismatchForUnloadableOrderIsRejected() {
        TestPaymentService service = configuredAlipay(mock(AlipayClient.class));
        when(orderMapper.selectById(99L)).thenReturn(null);

        assertThrows(IllegalStateException.class, () -> service.getAlipayParams(99L, 1L));
    }

    @Test
    void stripeAmountIsConvertedToMinorUnits() {
        TestPaymentService service = service(mock(AlipayClient.class));
        assertEquals(1234L, service.toMinorUnits(new BigDecimal("12.34")));
        assertEquals(100L, service.toMinorUnits(new BigDecimal("1")));
        assertThrows(IllegalArgumentException.class, () -> service.toMinorUnits(BigDecimal.ZERO));
    }

    static class FakeOrder {
        private final Long id;
        private final BigDecimal amount;

        FakeOrder(Long id, BigDecimal amount) {
            this.id = id;
            this.amount = amount;
        }

        public Long getId() {
            return id;
        }

        public BigDecimal getAmount() {
            return amount;
        }
    }
}
