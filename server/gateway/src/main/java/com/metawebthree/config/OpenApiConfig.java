package com.metawebthree.config;

import java.util.ArrayList;
import java.net.HttpURLConnection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springdoc.core.models.GroupedOpenApi;
import org.springdoc.core.properties.SwaggerUiConfigProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.DiscoveryClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

import com.metawebthree.common.cloud.DiscoveryClientSupport;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;

@Slf4j
@Configuration
@RequiredArgsConstructor
public class OpenApiConfig {

    @Value("${spring.application.name}")
    private String applicationName;
    private final DiscoveryClient discoveryClient;
    private final SwaggerUiConfigProperties swaggerUiConfigProperties;
    private final WebClient.Builder webClientBuilder;
    private final ObjectMapper objectMapper;
    
    private volatile Map<String, JsonNode> cachedApiDocs = new HashMap<>();

    @Bean
    public org.springdoc.core.models.GroupedOpenApi allGroupedOpenApi() {
        return org.springdoc.core.models.GroupedOpenApi.builder()
                .group("all")
                .pathsToMatch("/**")
                .build();
    }

    @Scheduled(fixedRate = 30 * 1000, initialDelay = 1000)
    public void updateSwaggerUrls() {
        Set<SwaggerUiConfigProperties.SwaggerUrl> swaggerUrls = new HashSet<>();
        List<String> services = DiscoveryClientSupport.getServicesSafely(discoveryClient, applicationName);

        for (String serviceId : services) {
            if (applicationName.equalsIgnoreCase(serviceId)) {
                continue;
            }
            try {
                List<ServiceInstance> instances = DiscoveryClientSupport.getInstancesSafely(
                        discoveryClient,
                        serviceId,
                        applicationName);
                if (instances.isEmpty() || instances.size() == 0) {
                    continue;
                }
                ServiceInstance instance = instances.get(0);
                boolean docsAvailable = testApiDocs(instance);
                if (!docsAvailable) {
                    continue;
                }
                SwaggerUiConfigProperties.SwaggerUrl swaggerUrl = new SwaggerUiConfigProperties.SwaggerUrl();
                swaggerUrl.setName(serviceId);
                swaggerUrl.setUrl("/" + serviceId + "/v3/api-docs");
                swaggerUrls.add(swaggerUrl);
            } catch (Exception e) {
                continue;
            }
        }
        swaggerUiConfigProperties.setUrls(swaggerUrls);
    }

    @Scheduled(fixedRate = 60 * 1000, initialDelay = 2000)
    public void cacheApiDocs() {
        Map<String, JsonNode> newCache = new HashMap<>();
        List<String> services = DiscoveryClientSupport.getServicesSafely(discoveryClient, applicationName);
        log.info("Caching API docs for {} services: {}", services.size(), services);
        
        for (String serviceId : services) {
            if (applicationName.equalsIgnoreCase(serviceId)) {
                continue;
            }
            try {
                List<ServiceInstance> instances = DiscoveryClientSupport.getInstancesSafely(
                        discoveryClient,
                        serviceId,
                        applicationName);
                if (instances.isEmpty()) {
                    log.warn("No instances found for service: {}", serviceId);
                    continue;
                }
                ServiceInstance instance = instances.get(0);
                JsonNode apiDocs = fetchApiDocs(instance);
                if (apiDocs != null) {
                    int schemaCount = apiDocs.has("components") && apiDocs.get("components").has("schemas") 
                        ? apiDocs.get("components").get("schemas").size() : 0;
                    log.info("Fetched {} schemas from service {}", schemaCount, serviceId);
                    newCache.put(serviceId, apiDocs);
                }
            } catch (Exception e) {
                log.warn("Failed to fetch API docs for service: {}", serviceId, e);
            }
        }
        
        cachedApiDocs = newCache;
        log.info("Cached API docs for {} services", cachedApiDocs.size());
        for (Map.Entry<String, JsonNode> entry : cachedApiDocs.entrySet()) {
            int schemaCount = entry.getValue().has("components") && entry.getValue().get("components").has("schemas")
                ? entry.getValue().get("components").get("schemas").size() : 0;
            log.info("  - {}: {} schemas", entry.getKey(), schemaCount);
        }
    }

    private boolean testApiDocs(ServiceInstance instance) {
        String apiDocsUrl = instance.getUri() + "/v3/api-docs";
        WebClient webClient = webClientBuilder.build();
        try {
            HttpStatusCode statusCode = webClient.get()
                    .uri(apiDocsUrl)
                    .retrieve()
                    .toBodilessEntity()
                    .map(response -> response.getStatusCode())
                    .onErrorResume(e -> Mono.just(HttpStatusCode.valueOf(500)))
                    .block();
            if (statusCode == null) {
                return false;
            }
            return statusCode.is2xxSuccessful();
        } catch (Exception e) {
            return false;
        }
    }

    private JsonNode fetchApiDocs(ServiceInstance instance) {
        String apiDocsUrl = instance.getUri() + "/v3/api-docs";
        WebClient webClient = webClientBuilder.build();
        try {
            String response = webClient.get()
                    .uri(apiDocsUrl)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();
            return objectMapper.readTree(response);
        } catch (Exception e) {
            log.warn("Failed to fetch API docs from: {}", apiDocsUrl, e);
            return null;
        }
    }

    @Slf4j
    @RestController
    public static class AggregatedApiDocsController {
        
        private final OpenApiConfig openApiConfig;
        private final ObjectMapper objectMapper;
        
        public AggregatedApiDocsController(OpenApiConfig openApiConfig, ObjectMapper objectMapper) {
            this.openApiConfig = openApiConfig;
            this.objectMapper = objectMapper;
        }
        
        @GetMapping(value = "/pump/v3/api-docs", produces = MediaType.APPLICATION_JSON_VALUE)
        public String getAggregatedApiDocs() {
            try {
                List<String> services = DiscoveryClientSupport.getServicesSafely(openApiConfig.discoveryClient, openApiConfig.applicationName);
                log.info("Found {} services: {}", services.size(), services);
                Map<String, JsonNode> docs = new HashMap<>();
                
                for (String serviceId : services) {
                    if (openApiConfig.applicationName.equalsIgnoreCase(serviceId)) continue;
                    try {
                        List<ServiceInstance> instances = DiscoveryClientSupport.getInstancesSafely(
                                openApiConfig.discoveryClient, serviceId, openApiConfig.applicationName);
                        if (instances.isEmpty()) {
                            log.warn("No instances for service: {}", serviceId);
                            continue;
                        }
                        
                        String apiDocsUrl = instances.get(0).getUri() + "/v3/api-docs";
                        log.info("Fetching {} for service {}", apiDocsUrl, serviceId);
                        String response = openApiConfig.webClientBuilder.build().get()
                                .uri(apiDocsUrl)
                                .retrieve()
                                .bodyToMono(String.class)
                                .block();
                        
                        JsonNode doc = objectMapper.readTree(response);
                        int schemaCount = doc.has("components") && doc.get("components").has("schemas")
                            ? doc.get("components").get("schemas").size() : 0;
                        log.info("Fetched {} schemas from service {}", schemaCount, serviceId);
                        docs.put(serviceId, doc);
                    } catch (Exception e) {
                        log.warn("Failed to fetch API docs for service {}: {}", serviceId, e.getMessage());
                    }
                }
                
                log.info("Collected docs from {} services", docs.size());
                if (docs.isEmpty()) {
                    return createEmptyApiDocs();
                }
                
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
        
        private void renameSchemaRefs(JsonNode node, String serviceName, ObjectNode schemas, Map<String, JsonNode> docs) {
            if (node == null) return;
            
            if (node.isObject()) {
                ObjectNode obj = (ObjectNode) node;
                
                if (obj.has("$ref")) {
                    String ref = obj.get("$ref").asText();
                    if (ref.startsWith("#/components/schemas/")) {
                        String schemaName = ref.substring("#/components/schemas/".length());
                        String prefixedName = serviceName + "_" + schemaName;
                        obj.put("$ref", "#/components/schemas/" + prefixedName);
                        
                        JsonNode originalSchema = null;
                        for (Map.Entry<String, JsonNode> docEntry : docs.entrySet()) {
                            JsonNode doc = docEntry.getValue();
                            if (doc.has("components") && doc.get("components").has("schemas")) {
                                JsonNode schema = doc.get("components").get("schemas").get(schemaName);
                                if (schema != null) {
                                    originalSchema = schema;
                                    break;
                                }
                            }
                        }
                        
                        if (originalSchema != null && !schemas.has(prefixedName)) {
                            schemas.set(prefixedName, originalSchema.deepCopy());
                        } else if (originalSchema == null) {
                            log.warn("Schema {} not found in any service", schemaName);
                        }
                        return;
                    }
                }
                
                if (obj.has("allOf")) {
                    obj.fieldNames().forEachRemaining(field -> {
                        if (!"$ref".equals(field)) {
                            renameSchemaRefs(obj.get(field), serviceName, schemas, docs);
                        }
                    });
                } else if (obj.has("oneOf")) {
                    obj.fieldNames().forEachRemaining(field -> {
                        if (!"$ref".equals(field)) {
                            renameSchemaRefs(obj.get(field), serviceName, schemas, docs);
                        }
                    });
                } else if (obj.has("anyOf")) {
                    obj.fieldNames().forEachRemaining(field -> {
                        if (!"$ref".equals(field)) {
                            renameSchemaRefs(obj.get(field), serviceName, schemas, docs);
                        }
                    });
                } else if (obj.has("items")) {
                    renameSchemaRefs(obj.get("items"), serviceName, schemas, docs);
                } else if (obj.has("properties")) {
                    obj.get("properties").fieldNames().forEachRemaining(prop -> {
                        renameSchemaRefs(obj.get("properties").get(prop), serviceName, schemas, docs);
                    });
                } else {
                    obj.fieldNames().forEachRemaining(field -> {
                        renameSchemaRefs(obj.get(field), serviceName, schemas, docs);
                    });
                }
            } else if (node.isArray()) {
                for (JsonNode item : node) {
                    renameSchemaRefs(item, serviceName, schemas, docs);
                }
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
}
