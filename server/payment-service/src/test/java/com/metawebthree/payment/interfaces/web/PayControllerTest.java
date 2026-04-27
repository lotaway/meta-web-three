package com.metawebthree.payment.interfaces.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.payment.application.PaymentService;

class PayControllerTest {

    @Mock
    private PaymentService paymentService;

    private PayController payController;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        payController = new PayController(paymentService);
    }

    @Test
    void getWechatParamsReturnsParamMissingErrorWhenOrderIdMissing() {
        ApiResponse<Map<String, String>> response = payController.getWechatParams(1L, Map.of());

        assertEquals("1002", response.getCode());
        assertNull(response.getData());
    }

    @Test
    void verifyPaymentReturnsValidFlagFromService() {
        when(paymentService.verifyPayment("1001", "PREPAY_1001", 1L)).thenReturn(true);

        ApiResponse<Map<String, Object>> response = payController.verifyPayment(
                1L,
                Map.of("orderId", "1001", "transactionId", "PREPAY_1001"));

        assertEquals("0000", response.getCode());
        assertTrue((Boolean) response.getData().get("valid"));
    }

    @Test
    void verifyPaymentReturnsFalseWhenServiceRejectsTransaction() {
        when(paymentService.verifyPayment("1001", "bad", 1L)).thenReturn(false);

        ApiResponse<Map<String, Object>> response = payController.verifyPayment(
                1L,
                Map.of("orderId", "1001", "transactionId", "bad"));

        assertEquals("0000", response.getCode());
        assertFalse((Boolean) response.getData().get("valid"));
    }
}
