package com.metawebthree.Controller;

import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

import reactor.core.publisher.Mono;

import org.springframework.web.bind.annotation.GetMapping;

@RestController
public class SwaggerController {
    private final WebClient.Builder webClientBuilder;

    public SwaggerController(WebClient.Builder webClientBuilder) {
        this.webClientBuilder = webClientBuilder;
    }

    @GetMapping("/v3/api-docs/{service}")
    public Mono<String> getServiceDocs(@PathVariable String service) {
        String url = switch (service) {
            case "user" -> "http://user-service/v3/api-docs";
            case "order" -> "http://order-service/v3/api-docs";
            case "product" -> "http://product-service/v3/api-docs";
            default -> throw new RuntimeException("service not found: " + service);
        };
        return webClientBuilder.build()
                .get()
                .uri(url)
                .retrieve()
                .bodyToMono(String.class);
    }
}
