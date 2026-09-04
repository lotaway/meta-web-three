package com.metawebthree.recommendation.domain.aishopping.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.application.aishopping.AiProviderConfig;
import com.metawebthree.recommendation.application.aishopping.AiProviderSettings;
import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductDataProvider;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.InMemoryVectorStore;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorStoreFactory;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AiShoppingVectorSearchTest {

    private VectorStoreFactory vectorStoreFactory;
    private ProductCatalogCache catalogCache;
    private AiProviderConfig providerConfig;
    private AiShoppingVectorSearch service;

    @BeforeEach
    void setUp() {
        vectorStoreFactory = mock(VectorStoreFactory.class);
        catalogCache = mock(ProductCatalogCache.class);
        providerConfig = mock(AiProviderConfig.class);

        AiProviderSettings settings = mock(AiProviderSettings.class);
        when(providerConfig.getSettings()).thenReturn(settings);
        when(settings.getMilvusCollectionText()).thenReturn("product_text");
        when(settings.getMilvusCollectionImage()).thenReturn("product_image");
        when(settings.getEmbeddingDim()).thenReturn(4);

        InMemoryVectorStore store = new InMemoryVectorStore();
        store.ensureCollection("product_text", 4);
        store.upsert("product_text", List.of(
                new com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorRecord(
                        1, 101, new float[]{1, 0, 0, 0}, java.util.Map.of()),
                new com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorRecord(
                        2, 999, new float[]{0, 1, 0, 0}, java.util.Map.of())));

        when(vectorStoreFactory.getStore()).thenReturn(store);

        ProductDataProvider.ProductItem product = new ProductDataProvider.ProductItem();
        product.id = 101;
        product.name = "智能手机 5G";
        product.pic = "http://img/1.jpg";
        product.price = 1999.0;
        when(catalogCache.get(101L)).thenReturn(product);

        service = new AiShoppingVectorSearch(vectorStoreFactory, catalogCache, providerConfig);
    }

    @Test
    void searchText_shouldReturnHydratedMatches() {
        List<AiProductMatch> matches = service.searchText(new float[]{1, 0, 0, 0}, 5);

        assertEquals(1, matches.size());
        AiProductMatch match = matches.get(0);
        assertEquals(101L, match.getProductId());
        assertEquals("智能手机 5G", match.getName());
        assertEquals("1999.0", match.getPrice());
        assertTrue(match.getScore() > 0.99);
    }

    @Test
    void searchText_shouldSkipProductsMissingFromCatalog() {
        List<AiProductMatch> matches = service.searchText(new float[]{0, 1, 0, 0}, 1);

        assertEquals(0, matches.size());
    }
}
