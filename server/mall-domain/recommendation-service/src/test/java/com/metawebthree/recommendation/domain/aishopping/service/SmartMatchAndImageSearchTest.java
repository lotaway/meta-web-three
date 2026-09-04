package com.metawebthree.recommendation.domain.aishopping.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SmartMatchAndImageSearchTest {

    private AiProviderClient providerClient;
    private AiShoppingVectorSearch vectorSearch;
    private SmartMatchService smartMatchService;
    private ImageSearchService imageSearchService;

    @BeforeEach
    void setUp() {
        providerClient = mock(AiProviderClient.class);
        vectorSearch = mock(AiShoppingVectorSearch.class);
        smartMatchService = new SmartMatchService(providerClient, vectorSearch);
        imageSearchService = new ImageSearchService(providerClient, vectorSearch);
    }

    @Test
    void smartMatch_shouldEmbedAndSearch() {
        when(providerClient.embedText("蓝牙耳机")).thenReturn(new float[]{1, 0, 0});
        AiProductMatch match = new AiProductMatch(1L, "蓝牙耳机", "http://img/x.jpg", "99.0", 0.95, "Smart match similarity 0.95");
        when(vectorSearch.searchText(any(float[].class), eq(5))).thenReturn(List.of(match));

        List<AiProductMatch> result = smartMatchService.match("蓝牙耳机", 5);

        assertEquals(1, result.size());
        assertEquals("蓝牙耳机", result.get(0).getName());
        verify(providerClient).embedText("蓝牙耳机");
    }

    @Test
    void imageSearch_shouldEmbedImageAndSearch() {
        when(providerClient.embedImage(any(byte[].class))).thenReturn(new float[]{0, 1, 0});
        AiProductMatch match = new AiProductMatch(2L, "无线耳机", "http://img/y.jpg", "199.0", 0.88, "Image search similarity 0.88");
        when(vectorSearch.searchImage(any(float[].class), eq(10))).thenReturn(List.of(match));

        List<AiProductMatch> result = imageSearchService.search(new byte[]{1, 2, 3}, 10);

        assertEquals(1, result.size());
        assertEquals("无线耳机", result.get(0).getName());
        verify(providerClient).embedImage(any(byte[].class));
    }

    @Test
    void smartMatch_shouldPropagateProviderError() {
        when(providerClient.embedText(anyString()))
                .thenThrow(new IllegalStateException("embedding provider not configured"));

        assertThrows(IllegalStateException.class, () -> smartMatchService.match("耳机", 5));
    }
}
