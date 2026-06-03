package com.metawebthree.gateway.auth;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * API Key Authentication Filter for Gateway
 * Validates API keys for third-party developer access
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ApiKeyAuthFilter implements GlobalFilter, Ordered {

    private final ObjectMapper objectMapper;
    private final WebClient.Builder webClientBuilder;
    
    @Value("${gateway.api-key.header-key:X-API-Key}")
    private String apiKeyHeader;
    
    @Value("${gateway.api-key.header-secret:X-API-Secret}")
    private String apiSecretHeader;
    
    @Value("${gateway.api-key.signature-enabled:true}")
    private boolean signatureVerificationEnabled;
    
    @Value("${gateway.api-key.enabled:true}")
    private boolean apiKeyAuthEnabled;
    
    @Value("${gateway.developer-portal.url:http://localhost:8080}")
    private String developerPortalUrl;
    
    // Paths that require API key authentication
    private final List<String> protectedPaths = List.of(
        "/api/open/",
        "/developer/api/",
        "/open-api/"
    );
    
    // Paths that should skip API key authentication
    private final List<String> publicPaths = List.of(
        "/developer/register",
        "/developer/docs",
        "/oauth/authorize",
        "/oauth/token",
        "/actuator/health"
    );

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        if (!apiKeyAuthEnabled) {
            return chain.filter(exchange);
        }
        
        String path = exchange.getRequest().getPath().value();
        
        // Skip authentication for non-protected paths
        if (!requiresApiKeyAuth(path)) {
            return chain.filter(exchange);
        }
        
        // Skip public paths
        if (isPublicPath(path)) {
            return chain.filter(exchange);
        }
        
        // Extract API key and secret
        HttpHeaders headers = exchange.getRequest().getHeaders();
        String apiKeyId = headers.getFirst(apiKeyHeader);
        String apiSecret = headers.getFirst(apiSecretHeader);
        
        if (apiKeyId == null || apiSecret == null) {
            return writeErrorResponse(exchange, HttpStatus.UNAUTHORIZED, 
                "MISSING_API_CREDENTIALS", 
                "API key and secret are required. Include " + apiKeyHeader + " and " + apiSecretHeader + " headers.");
        }
        
        // Verify request signature (prevent tampering)
        if (signatureVerificationEnabled) {
            boolean signatureValid = verifyRequestSignature(exchange.getRequest(), apiKeyId, apiSecret);
            if (!signatureValid) {
                log.warn("Invalid request signature: keyId={}, path={}", apiKeyId, path);
                return writeErrorResponse(exchange, HttpStatus.UNAUTHORIZED, 
                    "INVALID_SIGNATURE", 
                    "Request signature verification failed. Possible request tampering detected.");
            }
        }
        
        // Validate API key by calling developer-portal-service
        Boolean isValid = isValidApiKey(apiKeyId, apiSecret).block();
        if (isValid == null || !isValid) {
            return writeErrorResponse(exchange, HttpStatus.UNAUTHORIZED, 
                "INVALID_API_CREDENTIALS", 
                "Invalid API key or secret");
        }
        
        // Add developer context to request headers
        ServerHttpRequest mutatedRequest = exchange.getRequest().mutate()
            .header("X-Developer-Id", extractDeveloperId(apiKeyId))
            .header("X-API-Key-Id", apiKeyId)
            .build();
        
        log.debug("API Key authenticated: keyId={}, path={}", apiKeyId, path);
        
        return chain.filter(exchange.mutate().request(mutatedRequest).build());
    }

    @Override
    public int getOrder() {
        return -90; // Run after authentication filter but before rate limiting
    }
    
    /**
     * Check if path requires API key authentication
     */
    private boolean requiresApiKeyAuth(String path) {
        return protectedPaths.stream().anyMatch(path::startsWith);
    }
    
    /**
     * Check if path is public (no auth required)
     */
    private boolean isPublicPath(String path) {
        return publicPaths.stream().anyMatch(path::startsWith);
    }
    
    /**
     * Validate API key by calling developer-portal-service
     */
    private Mono<Boolean> isValidApiKey(String apiKeyId, String apiSecret) {
        WebClient client = webClientBuilder.build();
        
        return client.get()
            .uri(developerPortalUrl + "/developer/api-keys/" + apiKeyId + "/validate")
            .header(apiSecretHeader, apiSecret)
            .retrieve()
            .bodyToMono(Map.class)
            .map(response -> {
                Boolean valid = (Boolean) response.get("valid");
                return valid != null && valid;
            })
            .onErrorReturn(false)
            .switchIfEmpty(Mono.just(false));
    }
    
    /**
     * Verify request signature to prevent tampering
     */
    private boolean verifyRequestSignature(ServerHttpRequest request, String apiKeyId, String apiSecret) {
        HttpHeaders headers = request.getHeaders();
        
        // Extract signature parameters
        String timestampStr = headers.getFirst("X-Timestamp");
        String nonce = headers.getFirst("X-Nonce");
        String clientSignature = headers.getFirst("X-Signature");
        
        if (timestampStr == null || nonce == null || clientSignature == null) {
            log.warn("Missing signature headers: timestamp={}, nonce={}, signature={}", 
                     timestampStr != null, nonce != null, clientSignature != null);
            return false;
        }
        
        try {
            long timestamp = Long.parseLong(timestampStr);
            
            // Extract request parameters
            Map<String, String> params = new HashMap<>();
            
            // Add query parameters
            request.getQueryParams().forEach((key, values) -> {
                if (!values.isEmpty()) {
                    params.put(key, values.get(0));
                }
            });
            
            // Generate expected signature
            String method = request.getMethod().name();
            String path = request.getPath().value();
            String expectedSignature = SignatureUtil.generateSignature(
                method, path, params, timestamp, nonce, apiSecret);
            
            // Verify signature (constant-time comparison)
            boolean valid = SignatureUtil.verifySignature(
                method, path, params, timestamp, nonce, apiSecret, clientSignature);
            
            if (valid) {
                log.debug("Signature verified successfully: keyId={}, path={}", apiKeyId, path);
            } else {
                log.warn("Signature mismatch: expected={}, actual={}", expectedSignature, clientSignature);
            }
            
            return valid;
            
        } catch (NumberFormatException e) {
            log.warn("Invalid timestamp format: {}", timestampStr);
            return false;
        } catch (Exception e) {
            log.error("Signature verification error", e);
            return false;
        }
    }
    
    /**
     * Extract developer ID from API key by calling developer-portal-service
     */
    private String extractDeveloperId(String apiKeyId) {
        try {
            WebClient client = webClientBuilder.build();
            Map response = client.get()
                .uri(developerPortalUrl + "/developer/api-keys/" + apiKeyId)
                .retrieve()
                .bodyToMono(Map.class)
                .block();
            
            if (response != null && response.containsKey("developerId")) {
                return (String) response.get("developerId");
            }
        } catch (Exception e) {
            log.warn("Failed to extract developer ID for key: {}", apiKeyId, e);
        }
        
        // Fallback: return placeholder
        return "dev_" + apiKeyId.hashCode();
    }
    
    /**
     * Write error response
     */
    private Mono<Void> writeErrorResponse(ServerWebExchange exchange, 
                                          HttpStatus status, 
                                          String code, 
                                          String message) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(status);
        response.getHeaders().setContentType(MediaType.APPLICATION_JSON);
        
        Map<String, Object> error = Map.of(
            "code", code,
            "message", message,
            "path", exchange.getRequest().getPath().value(),
            "timestamp", LocalDateTime.now().toString()
        );
        
        try {
            byte[] bytes = objectMapper.writeValueAsBytes(error);
            DataBuffer buffer = response.bufferFactory().wrap(bytes);
            return response.writeWith(Mono.just(buffer));
        } catch (Exception e) {
            log.error("Failed to write error response", e);
            return response.setComplete();
        }
    }
}
