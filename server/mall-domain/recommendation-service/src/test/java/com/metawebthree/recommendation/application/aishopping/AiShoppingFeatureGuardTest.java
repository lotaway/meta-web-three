package com.metawebthree.recommendation.application.aishopping;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.metawebthree.common.exception.BusinessException;
import org.junit.jupiter.api.Test;

class AiShoppingFeatureGuardTest {

    @Test
    void requireEnabled_whenDisabled_shouldThrow() {
        AiProviderConfig providerConfig = mock(AiProviderConfig.class);
        when(providerConfig.isEnabled()).thenReturn(false);
        AiShoppingFeatureGuard guard = new AiShoppingFeatureGuard(providerConfig);

        assertThrows(BusinessException.class, guard::requireEnabled);
    }

    @Test
    void requireEnabled_whenEnabled_shouldPass() {
        AiProviderConfig providerConfig = mock(AiProviderConfig.class);
        when(providerConfig.isEnabled()).thenReturn(true);
        AiShoppingFeatureGuard guard = new AiShoppingFeatureGuard(providerConfig);

        assertDoesNotThrow(guard::requireEnabled);
    }
}
