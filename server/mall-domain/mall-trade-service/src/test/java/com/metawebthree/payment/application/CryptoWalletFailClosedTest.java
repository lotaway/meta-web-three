package com.metawebthree.payment.application;

import com.metawebthree.payment.domain.exception.ExternalServiceException;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import org.junit.jupiter.api.Test;
import org.springframework.web.client.RestTemplate;

import java.lang.reflect.Field;
import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;

/**
 * The crypto wallet path must fail closed when it is not configured, rather than
 * returning a fabricated transaction hash.
 */
class CryptoWalletFailClosedTest {

    private final RestTemplate restTemplate = org.mockito.Mockito.mock(RestTemplate.class);

    private static void setField(Object target, String name, Object value) {
        try {
            Field field = CryptoWalletServiceImpl.class.getDeclaredField(name);
            field.setAccessible(true);
            field.set(target, value);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set field " + name, e);
        }
    }

    private static ExchangeOrder order(String currency, String amount, String toAddress) {
        ExchangeOrder order = new ExchangeOrder();
        order.setCryptoCurrency(currency);
        order.setCryptoAmount(new BigDecimal(amount));
        order.setUserWalletAddress(toAddress);
        return order;
    }

    @Test
    void transferRefusesToFakeWhenNoWalletIntegrationConfigured() {
        CryptoWalletServiceImpl service = new CryptoWalletServiceImpl(restTemplate);

        assertEquals(BigDecimal.ZERO, service.getWalletBalance("BTC"));
        assertThrows(ExternalServiceException.class,
                () -> service.transferCrypto(order("BTC", "1.0", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")));
    }

    @Test
    void transferRefusesWhenHotWalletAddressMissing() {
        CryptoWalletServiceImpl service = spy(new CryptoWalletServiceImpl(restTemplate));
        doReturn(new BigDecimal("10")).when(service).getWalletBalance("BTC");
        setField(service, "blockchainApiUrl", "https://wallet.example");

        ExternalServiceException ex = assertThrows(ExternalServiceException.class,
                () -> service.transferCrypto(order("BTC", "1.0", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")));

        assertTrue(ex.getMessage().contains("No hot wallet address configured"), ex.getMessage());
    }
}
