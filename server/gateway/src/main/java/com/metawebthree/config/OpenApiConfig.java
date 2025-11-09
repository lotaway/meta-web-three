package com.metawebthree.config;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.springdoc.core.models.GroupedOpenApi;
import org.springdoc.core.properties.SwaggerUiConfigProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.ServiceInstance;
import org.springframework.cloud.client.discovery.DiscoveryClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.reactive.function.client.WebClient;

import lombok.RequiredArgsConstructor;
import reactor.core.publisher.Mono;

@Configuration
@RequiredArgsConstructor
public class OpenApiConfig {

    @Value("${spring.application.name}")
    private String applicationName;
    private final DiscoveryClient discoveryClient;
    private final SwaggerUiConfigProperties swaggerUiConfigProperties;
    private final WebClient.Builder webClientBuilder;

    @Bean
    public GroupedOpenApi allGroupedOpenApi() {
        return GroupedOpenApi.builder()
                .group("all")
                .pathsToMatch("/**")
                .build();
    }

    @Scheduled(fixedRate = 30 * 1000, initialDelay = 1000)
    public void updateSwaggerUrls() {
        Set<SwaggerUiConfigProperties.SwaggerUrl> swaggerUrls = new HashSet<>();
        List<String> services = discoveryClient.getServices();

        for (String serviceId : services) {
            if (applicationName.equalsIgnoreCase(serviceId)) {
                continue;
            }
            List<ServiceInstance> instances = discoveryClient.getInstances(serviceId);
            if (instances.isEmpty()) {
                continue;
            }
            ServiceInstance instance = instances.get(0);
            boolean healthy = testHealthy(instance);
            if (!healthy) {
                continue;
            }

            SwaggerUiConfigProperties.SwaggerUrl swaggerUrl = new SwaggerUiConfigProperties.SwaggerUrl();
            swaggerUrl.setName(serviceId);
            swaggerUrl.setUrl("/" + serviceId + "/v3/api-docs");
            swaggerUrls.add(swaggerUrl);
        }
        swaggerUiConfigProperties.setUrls(swaggerUrls);
    }

    public boolean testHealthy(ServiceInstance instance) {
        String healthUrl = instance.getUri() + "/actuator/health";
        WebClient webClient = webClientBuilder.build();
        try {
            HttpStatusCode statusCode = webClient.get()
                    .uri(healthUrl)
                    .retrieve()
                    .toBodilessEntity()
                    .map(response -> response.getStatusCode())
                    .onErrorResume(e -> Mono.just(HttpStatusCode.valueOf(HttpStatus.INTERNAL_SERVER_ERROR.value())))
                    .block();
            if (statusCode == null) {
                return false;
            }
            return statusCode.is2xxSuccessful();
        } catch (Exception e) {
            return false;
        }
    }
}