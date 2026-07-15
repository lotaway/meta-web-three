package com.metawebthree.config;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springdoc.core.properties.SwaggerUiConfigProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.DiscoveryClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatusCode;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.reactive.function.client.WebClient;

import com.metawebthree.common.cloud.DiscoveryClientSupport;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;

@Slf4j
@Configuration
public class OpenApiConfig {

    @Value("${spring.application.name}")
    private String applicationName;
    private final DiscoveryClient discoveryClient;
    private final SwaggerUiConfigProperties swaggerUiConfigProperties;
    private final WebClient.Builder webClientBuilder;
    private final ObjectMapper objectMapper;

    private volatile Map<String, JsonNode> cachedApiDocs = new HashMap<>();

    public Map<String, JsonNode> getCachedApiDocs() {
        return cachedApiDocs;
    }

    public OpenApiConfig(DiscoveryClient discoveryClient,
                         SwaggerUiConfigProperties swaggerUiConfigProperties,
                         WebClient.Builder webClientBuilder,
                         ObjectMapper objectMapper) {
        this.discoveryClient = discoveryClient;
        this.swaggerUiConfigProperties = swaggerUiConfigProperties;
        this.webClientBuilder = webClientBuilder;
        this.objectMapper = objectMapper;
    }

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
                            ? apiDocs.get("components").get("schemas").size()
                            : 0;
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
                    ? entry.getValue().get("components").get("schemas").size()
                    : 0;
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

}
