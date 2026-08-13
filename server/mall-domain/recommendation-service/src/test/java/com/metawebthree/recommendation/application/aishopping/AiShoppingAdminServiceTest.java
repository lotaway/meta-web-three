package com.metawebthree.recommendation.application.aishopping;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.aishopping.entity.AiShoppingConfig;
import com.metawebthree.recommendation.domain.aishopping.repository.AiSearchLogRepository;
import com.metawebthree.recommendation.domain.aishopping.repository.AiShoppingConfigRepository;
import com.metawebthree.recommendation.domain.aishopping.service.AiShoppingIndexService;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AiShoppingAdminServiceTest {

    private AiShoppingConfigRepository configRepository;
    private AiSearchLogRepository logRepository;
    private AiShoppingIndexService indexService;
    private AiProviderConfig providerConfig;
    private AiProviderClient providerClient;
    private AiShoppingAdminService service;

    @BeforeEach
    void setUp() {
        configRepository = mock(AiShoppingConfigRepository.class);
        logRepository = mock(AiSearchLogRepository.class);
        indexService = mock(AiShoppingIndexService.class);
        providerConfig = mock(AiProviderConfig.class);
        providerClient = mock(AiProviderClient.class);

        service = new AiShoppingAdminService(configRepository, logRepository, indexService,
                providerConfig, providerClient);
    }

    @Test
    void saveConfig_shouldPersist() {
        when(configRepository.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        AiShoppingConfig saved = service.saveConfig("llm.model", "qwen-max", "model");

        assertEquals("llm.model", saved.getConfigKey());
        assertEquals("qwen-max", saved.getConfigValue());
    }

    @Test
    void saveConfig_withBlankKey_shouldThrow() {
        assertThrows(IllegalArgumentException.class, () -> service.saveConfig("  ", "v", "d"));
    }

    @Test
    void testProvider_shouldReportSuccess() {
        when(providerClient.embedText(anyString())).thenReturn(new float[]{1, 0});

        Map<String, Object> result = service.testProvider("embedding");

        assertTrue((Boolean) result.get("success"));
    }

    @Test
    void testProvider_shouldReportFailure() {
        when(providerClient.embedText(anyString()))
                .thenThrow(new IllegalStateException("not configured"));

        Map<String, Object> result = service.testProvider("embedding");

        assertFalse((Boolean) result.get("success"));
        assertNotNull(result.get("error"));
    }

    @Test
    void indexStatus_shouldIncludeStoreAndCollections() {
        AiProviderSettings settings = mock(AiProviderSettings.class);
        when(providerConfig.getSettings()).thenReturn(settings);
        when(settings.getVectorStore()).thenReturn("milvus");
        when(settings.getMilvusCollectionText()).thenReturn("product_text");
        when(settings.getMilvusCollectionImage()).thenReturn("product_image");
        when(indexService.getStatus()).thenReturn(
                new com.metawebthree.recommendation.domain.aishopping.entity.IndexStatus());

        Map<String, Object> result = service.indexStatus();

        assertEquals("milvus", result.get("vectorStore"));
        assertEquals("product_text", result.get("collectionText"));
    }

    @Test
    void recentLogs_shouldDefaultLimit() {
        when(logRepository.findRecent(50)).thenReturn(List.of());

        assertTrue(service.recentLogs(0).isEmpty());
        verify(logRepository).findRecent(50);
    }
}
