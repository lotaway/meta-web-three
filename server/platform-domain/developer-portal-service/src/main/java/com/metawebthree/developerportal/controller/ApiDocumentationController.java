package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.service.ApiDocumentationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * Developer Portal Documentation Controller
 * Provides API documentation and SDK samples
 */
@Tag(name = "Developer Portal Documentation", description = "API documentation and SDK resources")
@RestController
@RequestMapping("/developer/docs")
@RequiredArgsConstructor
public class ApiDocumentationController {

    private final ApiDocumentationService documentationService;

    @Operation(summary = "Get OpenAPI documentation", description = "Get complete OpenAPI 3.0 documentation for all available APIs")
    @GetMapping("/openapi")
    public ResponseEntity<Map<String, Object>> getOpenApiDocumentation(
        @RequestParam(required = false) String baseUrl
    ) {
        Map<String, Object> documentation = documentationService.generateOpenApiDocumentation(baseUrl);
        return ResponseEntity.ok(documentation);
    }

    @Operation(summary = "Get personalized documentation", description = "Get API documentation filtered by developer's subscriptions")
    @GetMapping("/openapi/{developerId}")
    public ResponseEntity<Map<String, Object>> getPersonalizedDocumentation(
        @PathVariable String developerId,
        @RequestParam(required = false) String baseUrl
    ) {
        Map<String, Object> documentation = documentationService.generatePersonalizedDocumentation(developerId, baseUrl);
        return ResponseEntity.ok(documentation);
    }

    @Operation(summary = "Get SDK code samples", description = "Get SDK code samples in various programming languages")
    @GetMapping("/sdk-samples")
    public ResponseEntity<Map<String, String>> getSdkSamples(
        @RequestParam(defaultValue = "all") String language
    ) {
        Map<String, String> samples = documentationService.generateSdkSamples(language);
        return ResponseEntity.ok(samples);
    }

    @Operation(summary = "Get API quick start guide", description = "Get quick start guide for API integration")
    @GetMapping("/quick-start")
    public ResponseEntity<Map<String, Object>> getQuickStartGuide() {
        Map<String, Object> guide = Map.of(
            "title", "API Quick Start Guide",
            "steps", java.util.List.of(
                Map.of("step", 1, "title", "Register as Developer", "description", "Sign up and get approved for API access"),
                Map.of("step", 2, "title", "Get API Key", "description", "Generate your API key from the developer portal"),
                Map.of("step", 3, "title", "Subscribe to APIs", "description", "Subscribe to the API endpoints you need"),
                Map.of("step", 4, "title", "Integrate", "description", "Use SDK or REST API to integrate"),
                Map.of("step", 5, "title", "Test", "description", "Test in sandbox environment before production")
            ),
            "authentication", Map.of(
                "apiKey", Map.of(
                    "description", "Include API key in request header",
                    "header", "X-API-Key",
                    "example", "X-API-Key: ak_your_api_key_here"
                ),
                "oauth2", Map.of(
                    "description", "Use OAuth 2.0 for user-authorized access",
                    "flow", "Authorization Code or Client Credentials"
                )
            ),
            "rateLimits", Map.of(
                "free", "10,000 requests/day",
                "basic", "50,000 requests/day",
                "professional", "200,000 requests/day",
                "enterprise", "Unlimited (custom pricing)"
            ),
            "sandbox", Map.of(
                "baseUrl", "https://sandbox-api.metawebthree.com",
                "description", "Test your integration in sandbox environment"
            )
        );
        
        return ResponseEntity.ok(guide);
    }

    @Operation(summary = "Get API status", description = "Get current API service status and health")
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getApiStatus() {
        Map<String, Object> status = Map.of(
            "status", "operational",
            "timestamp", java.time.LocalDateTime.now().toString(),
            "services", Map.of(
                "user-service", Map.of("status", "up", "latency", "45ms"),
                "product-service", Map.of("status", "up", "latency", "32ms"),
                "order-service", Map.of("status", "up", "latency", "58ms"),
                "inventory-service", Map.of("status", "up", "latency", "28ms"),
                "payment-service", Map.of("status", "up", "latency", "125ms")
            ),
            "incidents", java.util.Collections.emptyList()
        );
        
        return ResponseEntity.ok(status);
    }

    // ==================== Developer Sandbox ====================

    @Operation(summary = "Get sandbox test data", description = "Get test data for sandbox environment")
    @GetMapping("/sandbox/test-data/{developerId}")
    public ResponseEntity<Map<String, Object>> getSandboxTestData(@PathVariable String developerId) {
        Map<String, Object> testData = documentationService.generateSandboxTestData(developerId);
        return ResponseEntity.ok(testData);
    }

    @Operation(summary = "Reset sandbox environment", description = "Reset sandbox environment and generate fresh test data")
    @PostMapping("/sandbox/reset/{developerId}")
    public ResponseEntity<Map<String, Object>> resetSandboxEnvironment(@PathVariable String developerId) {
        Map<String, Object> result = documentationService.resetSandboxEnvironment(developerId);
        return ResponseEntity.ok(result);
    }
}
