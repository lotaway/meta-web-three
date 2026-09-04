package com.metawebthree.recommendation.domain.aishopping.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductDataProvider;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LocalTextCorrectorTest {

    private ProductCatalogCache catalogCache;
    private LocalTextCorrector corrector;

    @BeforeEach
    void setUp() {
        catalogCache = mock(ProductCatalogCache.class);

        ProductDataProvider.ProductItem phone = new ProductDataProvider.ProductItem();
        phone.id = 1;
        phone.name = "智能手机 5G 手机";
        ProductDataProvider.ProductItem headphones = new ProductDataProvider.ProductItem();
        headphones.id = 2;
        headphones.name = "无线蓝牙耳机";
        ProductDataProvider.ProductItem laptop = new ProductDataProvider.ProductItem();
        laptop.id = 3;
        laptop.name = "笔记本电脑 轻薄本";

        when(catalogCache.all()).thenReturn(List.of(phone, headphones, laptop));
        corrector = new LocalTextCorrector(catalogCache);
    }

    @Test
    void suggest_shouldReturnSimilarCandidates() {
        List<String> suggestions = corrector.suggest("蓝牙耳机");

        assertFalse(suggestions.isEmpty());
        assertTrue(suggestions.stream().anyMatch(s -> s.contains("无线蓝牙耳机")));
    }

    @Test
    void bestCorrection_shouldReturnExactMatch() {
        assertEquals("智能手机 5g 手机", corrector.bestCorrection("智能手机 5G 手机"));
    }

    @Test
    void bestCorrection_shouldReturnNullForUnknownQuery() {
        assertNull(corrector.bestCorrection("完全无关词汇xyz"));
    }

    @Test
    void suggest_shouldHandleBlankQuery() {
        assertTrue(corrector.suggest("   ").isEmpty());
        assertNull(corrector.bestCorrection(""));
    }
}
