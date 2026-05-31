package com.metawebthree.Filters;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.TimeoutException;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import reactor.core.publisher.Mono;

/**
 * Gateway 熔断降级过滤器
 * 当下游服务不可用或响应过慢时，返回降级响应而不是错误
 * 使用 Spring Cloud Gateway 内置的响应式错误处理
 */
@Component
public class CircuitBreakerFilter implements GlobalFilter, Ordered {

    private final ObjectMapper objectMapper;
    
    // 降级响应码
    private static final String CODE_SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE";
    private static final String CODE_SERVICE_TIMEOUT = "SERVICE_TIMEOUT";
    private static final String CODE_SERVICE_ERROR = "SERVICE_ERROR";

    public CircuitBreakerFilter(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String path = exchange.getRequest().getPath().value();
        
        // 跳过不需要熔断的路径
        if (shouldSkipCircuitBreaker(path)) {
            return chain.filter(exchange);
        }
        
        String serviceName = determineServiceName(path);
        
        // 使用 onErrorResume 来捕获错误并返回降级响应
        return chain.filter(exchange)
                .onErrorResume(throwable -> handleFallback(exchange, throwable, serviceName));
    }

    private boolean shouldSkipCircuitBreaker(String path) {
        // 跳过健康检查和 actuator 端点
        return path.contains("/actuator") 
                || path.contains("/health")
                || path.contains("/v3/api-docs")
                || path.contains("/swagger-ui");
    }

    private String determineServiceName(String path) {
        if (path.contains("/product-service") || path.contains("/product/")) {
            return "productService";
        } else if (path.contains("/order-service") || path.contains("/order/")) {
            return "orderService";
        } else if (path.contains("/payment-service") || path.contains("/payment/")) {
            return "paymentService";
        } else if (path.contains("/user-service") || path.contains("/user/")) {
            return "userService";
        } else if (path.contains("/inventory-service") || path.contains("/inventory/")) {
            return "inventoryService";
        } else if (path.contains("/promotion-service") || path.contains("/promotion/")) {
            return "promotionService";
        } else if (path.contains("/cart-service") || path.contains("/cart/")) {
            return "cartService";
        } else if (path.contains("/warehouse-service") || path.contains("/warehouse/")) {
            return "warehouseService";
        } else if (path.contains("/logistics-service") || path.contains("/logistics/")) {
            return "logisticsService";
        } else if (path.contains("/finance-service") || path.contains("/finance/")) {
            return "financeService";
        } else if (path.contains("/settlement-service") || path.contains("/settlement/")) {
            return "settlementService";
        } else if (path.contains("/invoice-service") || path.contains("/invoice/")) {
            return "invoiceService";
        }
        return "backend";
    }

    private Mono<Void> handleFallback(ServerWebExchange exchange, Throwable throwable, String serviceName) {
        ServerHttpResponse response = exchange.getResponse();
        
        // 根据异常类型设置适当的 HTTP 状态码
        HttpStatus statusCode = determineHttpStatus(throwable);
        response.setStatusCode(statusCode);
        response.getHeaders().setContentType(MediaType.APPLICATION_JSON);

        String errorCode = determineErrorCode(throwable);
        String message = determineErrorMessage(throwable);
        
        Map<String, Object> payload = Map.of(
                "code", errorCode,
                "message", message,
                "service", serviceName,
                "path", exchange.getRequest().getPath().value(),
                "timestamp", Instant.now().toString());

        String json = toJson(payload);
        DataBuffer buffer = response.bufferFactory().wrap(json.getBytes(StandardCharsets.UTF_8));
        
        // 记录熔断日志
        System.err.println("[CircuitBreaker] Fallback triggered: service=" + serviceName + 
                          ", error=" + throwable.getClass().getSimpleName() + 
                          ", path=" + exchange.getRequest().getPath().value());
        
        return response.writeWith(Mono.just(buffer));
    }

    private HttpStatus determineHttpStatus(Throwable throwable) {
        if (throwable instanceof TimeoutException || 
            throwable.getCause() instanceof TimeoutException) {
            return HttpStatus.GATEWAY_TIMEOUT;
        } else if (throwable.getClass().getSimpleName().contains("CallNotPermittedException")) {
            return HttpStatus.SERVICE_UNAVAILABLE;
        }
        return HttpStatus.BAD_GATEWAY;
    }

    private String determineErrorCode(Throwable throwable) {
        String simpleName = throwable.getClass().getSimpleName().toLowerCase();
        if (simpleName.contains("timeout")) {
            return CODE_SERVICE_TIMEOUT;
        } else if (simpleName.contains("circuitbreaker") || 
                   simpleName.contains("callnotpermitted")) {
            return CODE_SERVICE_UNAVAILABLE;
        }
        return CODE_SERVICE_ERROR;
    }

    private String determineErrorMessage(Throwable throwable) {
        String simpleName = throwable.getClass().getSimpleName();
        String originalMessage = throwable.getMessage();
        
        if (simpleName.contains("TimeoutException")) {
            return "Service response timeout, please try again later";
        } else if (simpleName.contains("CallNotPermittedException")) {
            return "Service is temporarily unavailable (circuit breaker open)";
        } else if (simpleName.contains("BulkheadException")) {
            return "Service is busy, please try again later";
        } else if (simpleName.contains("ReadTimeoutException")) {
            return "Service read timeout, please try again later";
        } else if (simpleName.contains("ConnectException")) {
            return "Cannot connect to service, please check service status";
        }
        return "Service call failed" + (originalMessage != null ? ": " + originalMessage : "");
    }

    private String toJson(Map<String, Object> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException exception) {
            return "{\"code\":\"FALLBACK_ERROR\",\"message\":\"Service unavailable\"}";
        }
    }

    @Override
    public int getOrder() {
        // 在认证过滤器之后执行，但在其他业务逻辑之前
        return -90;
    }
}