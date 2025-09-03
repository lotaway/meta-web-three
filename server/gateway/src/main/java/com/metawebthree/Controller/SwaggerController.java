package com.metawebthree.Controller;

import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

import com.metawebthree.config.SwaggerUIProperties;
import com.metawebthree.config.SwaggerUIProperties.UrlConfig;

import reactor.core.publisher.Mono;
import org.springframework.web.bind.annotation.GetMapping;

// @RestController("/v3")
public class SwaggerController {
    private final WebClient.Builder webClientBuilder;

    private final SwaggerUIProperties swaggerUIProperties;

    public SwaggerController(WebClient.Builder webClientBuilder, SwaggerUIProperties swaggerUIProperties) {
        this.webClientBuilder = webClientBuilder;
        this.swaggerUIProperties = swaggerUIProperties;
    }

    @GetMapping("/api-docs/{serviceName}")
    public Mono<String> getServiceDocs(@PathVariable String service) {
        UrlConfig urlConfig = swaggerUIProperties.getUrlConfigs().stream().filter(uc -> uc.getName().equals(service)).findFirst().orElseThrow(() -> new RuntimeException("service not found: " + service));
        var sb = new StringBuilder();
        sb.append("http://").append(service).append(urlConfig.getUrl());
        String url = sb.toString();
        return webClientBuilder.build()
                .get()
                .uri(url)
                .retrieve()
                .bodyToMono(String.class);
    }
}
