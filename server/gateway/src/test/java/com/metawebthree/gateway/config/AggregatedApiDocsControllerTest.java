package com.metawebthree.gateway.config;

import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.config.OpenApiConfig;

public class AggregatedApiDocsControllerTest {

    private ObjectMapper objectMapper;
    private OpenApiConfig openApiConfig;
    private AggregatedApiDocsController controller;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        openApiConfig = Mockito.mock(OpenApiConfig.class);
        controller = new AggregatedApiDocsController(openApiConfig, objectMapper);
    }

    @Test
    public void shouldPrefixEveryPathWithItsServiceId() throws Exception {
        Mockito.when(openApiConfig.getCachedApiDocs()).thenReturn(Map.of(
                "mall-catalog-service", serviceDocs("/product/list", "/product-category/tree"),
                "mall-user-service", serviceDocs("/api/admin/user/list")));

        JsonNode paths = mergedPaths();

        Assertions.assertTrue(paths.has("/mall-catalog-service/product/list"));
        Assertions.assertTrue(paths.has("/mall-catalog-service/product-category/tree"));
        Assertions.assertTrue(paths.has("/mall-user-service/api/admin/user/list"));
    }

    @Test
    public void shouldNotLetOneServiceOverwriteAnotherServicePath() throws Exception {
        Mockito.when(openApiConfig.getCachedApiDocs()).thenReturn(Map.of(
                "mall-catalog-service", serviceDocs("/product/list"),
                "mall-trade-service", serviceDocs("/product/list")));

        JsonNode paths = mergedPaths();

        Assertions.assertTrue(paths.has("/mall-catalog-service/product/list"));
        Assertions.assertTrue(paths.has("/mall-trade-service/product/list"));
    }

    @Test
    public void shouldUseLowercaseServiceIdAsPrefix() throws Exception {
        Mockito.when(openApiConfig.getCachedApiDocs()).thenReturn(Map.of(
                "Mall-Catalog-Service", serviceDocs("/product/list")));

        JsonNode paths = mergedPaths();

        Assertions.assertTrue(paths.has("/mall-catalog-service/product/list"));
    }

    @Test
    public void shouldReturnEmptyDocumentWhenNoServiceIsRegistered() throws Exception {
        Mockito.when(openApiConfig.getCachedApiDocs()).thenReturn(Map.of());

        JsonNode merged = objectMapper.readTree(controller.getAggregatedApiDocs());

        Assertions.assertTrue(merged.get("paths").isEmpty());
    }

    private JsonNode mergedPaths() throws Exception {
        return objectMapper.readTree(controller.getAggregatedApiDocs()).get("paths");
    }

    private JsonNode serviceDocs(String... paths) throws Exception {
        StringBuilder json = new StringBuilder("{\"paths\":{");
        for (int i = 0; i < paths.length; i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append('"').append(paths[i]).append("\":{\"get\":{}}");
        }
        json.append("}}");
        return objectMapper.readTree(json.toString());
    }
}
