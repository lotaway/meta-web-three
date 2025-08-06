package com.metawebthree.service;

import com.metawebthree.dto.ExchangeOrderRequest;
import com.metawebthree.dto.ExchangeOrderResponse;
import com.metawebthree.entity.ExchangeOrder;
import com.metawebthree.entity.UserKYC;
import com.metawebthree.repository.ExchangeOrderRepository;
import com.metawebthree.repository.UserKYCRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ExchangeOrderServiceTest {

    @Mock
    private ExchangeOrderRepository exchangeOrderRepository;

    @Mock
    private UserKYCRepository userKYCRepository;

    @Mock
    private PriceEngineService priceEngineService;

    @Mock
    private RiskControlService riskControlService;

    @Mock
    private PaymentService paymentService;

    @Mock
    private CryptoWalletService cryptoWalletService;

    @InjectMocks
    private ExchangeOrderService exchangeOrderService;

    private ExchangeOrderRequest testRequest;
    private UserKYC testKYC;

    @BeforeEach
    void setUp() {
        testRequest = ExchangeOrderRequest.builder()
                .orderType("BUY_CRYPTO")
                .fiatCurrency("USD")
                .cryptoCurrency("BTC")
                .amount(new BigDecimal("1000"))
                .paymentMethod("ALIPAY")
                .walletAddress("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
                .autoExecute(true)
                .build();

        testKYC = UserKYC.builder()
                .userId(1L)
                .level(UserKYC.KYCLevel.L1)
                .status(UserKYC.KYCStatus.APPROVED)
                .build();
    }

    @Test
    void testCreateOrder_Success() {
        // Given
        when(userKYCRepository.findHighestApprovedLevelByUserId(1L))
                .thenReturn(Optional.of(testKYC));
        when(priceEngineService.getWeightedAveragePrice("BTC", "USD"))
                .thenReturn(new BigDecimal("45000"));
        when(exchangeOrderRepository.save(any(ExchangeOrder.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        // When
        ExchangeOrderResponse response = exchangeOrderService.createOrder(testRequest, 1L);

        // Then
        assertNotNull(response);
        assertEquals("PENDING", response.getStatus());
        assertEquals("BUY_CRYPTO", response.getOrderType());
        assertEquals("USD", response.getFiatCurrency());
        assertEquals("BTC", response.getCryptoCurrency());
        assertEquals(new BigDecimal("1000"), response.getFiatAmount());
        assertTrue(response.getCryptoAmount().compareTo(BigDecimal.ZERO) > 0);
    }

    @Test
    void testCreateOrder_KYCRequired() {
        // Given
        when(userKYCRepository.findHighestApprovedLevelByUserId(1L))
                .thenReturn(Optional.empty());

        // When & Then
        assertThrows(RuntimeException.class, () -> {
            exchangeOrderService.createOrder(testRequest, 1L);
        });
    }

    @Test
    void testCreateOrder_AmountExceedsLimit() {
        // Given
        testRequest.setAmount(new BigDecimal("50000")); // 超过L1级别限制
        when(userKYCRepository.findHighestApprovedLevelByUserId(1L))
                .thenReturn(Optional.of(testKYC));

        // When & Then
        assertThrows(RuntimeException.class, () -> {
            exchangeOrderService.createOrder(testRequest, 1L);
        });
    }
} 