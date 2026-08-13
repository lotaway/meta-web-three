package com.metawebthree.recommendation.domain.aishopping.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.aishopping.entity.TextCorrection;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductDataProvider;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TextCorrectionServiceTest {

    private AiProviderClient providerClient;
    private ProductCatalogCache catalogCache;
    private TextCorrectionService service;

    @BeforeEach
    void setUp() {
        providerClient = mock(AiProviderClient.class);
        catalogCache = mock(ProductCatalogCache.class);

        ProductDataProvider.ProductItem item = new ProductDataProvider.ProductItem();
        item.id = 1;
        item.name = "无线蓝牙耳机";
        when(catalogCache.all()).thenReturn(List.of(item));

        service = new TextCorrectionService(providerClient, catalogCache);
    }

    @Test
    void correct_shouldUseLlmWhenAvailable() {
        when(providerClient.chat(anyString(), anyString()))
                .thenReturn("{\"corrected\":\"蓝牙耳机\",\"suggestions\":[\"无线蓝牙耳机\"]}");

        TextCorrection result = service.correct("蓝牙耳ji");

        assertEquals("蓝牙耳机", result.getCorrected());
        assertTrue(result.isChanged());
        assertEquals(TextCorrection.CorrectionSource.LLM, result.getSource());
        assertTrue(result.getSuggestions().contains("无线蓝牙耳机"));
    }

    @Test
    void correct_shouldFallbackToLocalWhenLlmFails() {
        when(providerClient.chat(anyString(), anyString()))
                .thenThrow(new IllegalStateException("provider down"));

        TextCorrection result = service.correct("蓝牙耳机");

        assertEquals(TextCorrection.CorrectionSource.LOCAL, result.getSource());
        assertTrue(result.getCorrected().contains("无线蓝牙耳机"));
    }

    @Test
    void correct_shouldReturnNoChangeForBlank() {
        TextCorrection result = service.correct("  ");

        assertFalse(result.isChanged());
        assertEquals(TextCorrection.CorrectionSource.NONE, result.getSource());
    }
}
