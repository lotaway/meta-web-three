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

            ObjectNode merged = objectMapper.createObjectNode();
            merged.put("openapi", "3.1.0");

            ObjectNode info = merged.putObject("info");
            info.put("title", "Meta Web Three - Aggregated API");
            info.put("description", "Aggregated API documentation from all microservices");
            info.put("version", "1.0.0");

            ArrayNode servers = merged.putArray("servers");
            ObjectNode server = servers.addObject();
            server.put("url", "/");
            server.put("description", "Gateway");

            ObjectNode paths = merged.putObject("paths");
            ObjectNode components = merged.putObject("components");
            ObjectNode schemas = components.putObject("schemas");

            ArrayNode tags = merged.putArray("tags");

            for (Map.Entry<String, JsonNode> entry : docs.entrySet()) {
                String serviceName = entry.getKey();
                JsonNode serviceDocs = entry.getValue();

                ObjectNode tag = tags.addObject();
                tag.put("name", serviceName);
                tag.put("description", "Endpoints from " + serviceName);

                if (serviceDocs.has("components") && serviceDocs.get("components").has("schemas")) {
                    JsonNode serviceSchemas = serviceDocs.get("components").get("schemas");
                    serviceSchemas.fieldNames().forEachRemaining(schemaName -> {
                        if (!schemas.has(schemaName)) {
                            schemas.set(schemaName, serviceSchemas.get(schemaName).deepCopy());
                        }
                    });
                }

                if (serviceDocs.has("paths")) {
                    JsonNode servicePaths = serviceDocs.get("paths");
                    servicePaths.fieldNames().forEachRemaining(path -> {
                        if (!paths.has(path)) {
                            ((ObjectNode) paths).set(path, servicePaths.get(path).deepCopy());
                        }
                    });
                }
            }

            log.info("Aggregated docs: {} paths, {} schemas", paths.size(), schemas.size());
            String result = objectMapper.writeValueAsString(merged);
            log.info("Response size: {} bytes", result.length());
            return result;
        } catch (Exception e) {
            log.error("Failed to generate aggregated API docs", e);
            return createEmptyApiDocs();
        }
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
