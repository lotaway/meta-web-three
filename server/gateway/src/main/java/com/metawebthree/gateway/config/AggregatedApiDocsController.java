package com.metawebthree.gateway.config;

import java.util.Map;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import com.metawebthree.config.OpenApiConfig;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;

@Slf4j
@Configuration
public class AggregatedApiDocsController {

    private final OpenApiConfig openApiConfig;
    private final ObjectMapper objectMapper;

    public AggregatedApiDocsController(OpenApiConfig openApiConfig, ObjectMapper objectMapper) {
        this.openApiConfig = openApiConfig;
        this.objectMapper = objectMapper;
    }

    @Bean
    public RouterFunction<ServerResponse> apiDocsRoute() {
        return RouterFunctions.route()
            .GET("/meta/v3/api-docs", req ->
                ServerResponse.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Mono.fromCallable(this::getAggregatedApiDocs), String.class)
            )
            .build();
    }

    public String getAggregatedApiDocs() {
        try {
            Map<String, JsonNode> docs = openApiConfig.getCachedApiDocs();
            if (docs == null || docs.isEmpty()) {
                log.info("No cached API docs available, returning empty");
                return createEmptyApiDocs();
            }
            log.info("Returning cached API docs from {} services", docs.size());
            ObjectNode merged = createBaseDocument();
            ObjectNode paths = (ObjectNode) merged.get("paths");
            ObjectNode schemas = (ObjectNode) merged.get("components").get("schemas");
            ArrayNode tags = (ArrayNode) merged.get("tags");
            docs.forEach((serviceName, serviceDocs) -> {
                appendServiceTag(tags, serviceName);
                mergeServiceSchemas(schemas, serviceDocs);
                mergeServicePaths(paths, serviceName, serviceDocs);
            });
            log.info("Aggregated docs: {} paths, {} schemas", paths.size(), schemas.size());
            String result = objectMapper.writeValueAsString(merged);
            log.info("Response size: {} bytes", result.length());
            return result;
        } catch (Exception e) {
            log.error("Failed to generate aggregated API docs", e);
            return createEmptyApiDocs();
        }
    }

    private ObjectNode createBaseDocument() {
        ObjectNode merged = objectMapper.createObjectNode();
        merged.put("openapi", "3.1.0");
        ObjectNode info = merged.putObject("info");
        info.put("title", "Meta Web Three - Aggregated API");
        info.put("description", "Aggregated API documentation from all microservices");
        info.put("version", "1.0.0");
        ObjectNode server = merged.putArray("servers").addObject();
        server.put("url", "/");
        server.put("description", "Gateway");
        merged.putObject("paths");
        merged.putObject("components").putObject("schemas");
        merged.putArray("tags");
        return merged;
    }

    private void appendServiceTag(ArrayNode tags, String serviceName) {
        ObjectNode tag = tags.addObject();
        tag.put("name", serviceName);
        tag.put("description", "Endpoints from " + serviceName);
    }

    private void mergeServiceSchemas(ObjectNode schemas, JsonNode serviceDocs) {
        if (!serviceDocs.has("components") || !serviceDocs.get("components").has("schemas")) {
            return;
        }
        JsonNode serviceSchemas = serviceDocs.get("components").get("schemas");
        serviceSchemas.fieldNames().forEachRemaining(schemaName -> {
            if (!schemas.has(schemaName)) {
                schemas.set(schemaName, serviceSchemas.get(schemaName).deepCopy());
            }
        });
    }

    private void mergeServicePaths(ObjectNode paths, String serviceName, JsonNode serviceDocs) {
        if (!serviceDocs.has("paths")) {
            return;
        }
        JsonNode servicePaths = serviceDocs.get("paths");
        String servicePrefix = "/" + serviceName.toLowerCase();
        servicePaths.fieldNames().forEachRemaining(path -> {
            String prefixedPath = servicePrefix + (path.startsWith("/") ? path : "/" + path);
            if (!paths.has(prefixedPath)) {
                ((ObjectNode) paths).set(prefixedPath, servicePaths.get(path).deepCopy());
            }
        });
    }

    private String createEmptyApiDocs() {
        try {
            ObjectNode empty = objectMapper.createObjectNode();
            empty.put("openapi", "3.1.0");
            ObjectNode info = empty.putObject("info");
            info.put("title", "Meta Web Three - Aggregated API");
            info.put("version", "1.0.0");
            empty.putObject("paths");
            empty.putObject("components");
            return objectMapper.writeValueAsString(empty);
        } catch (Exception e) {
            return "{\"openapi\":\"3.1.0\",\"info\":{\"title\":\"Meta Web Three\",\"version\":\"1.0.0\"},\"paths\":{},\"components\":{}}";
        }
    }
}
