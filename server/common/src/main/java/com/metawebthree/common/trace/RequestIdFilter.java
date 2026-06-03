package com.metawebthree.common.trace;

import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.UUID;

/**
 * Filter to generate and propagate request ID across microservices
 * This enables distributed tracing across the entire call chain
 */
@Component
public class RequestIdFilter extends OncePerRequestFilter {
    
    public static final String REQUEST_ID_HEADER = "X-Request-ID";
    public static final String TRACE_ID_HEADER = "X-Trace-ID";
    
    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, 
                                    FilterChain filterChain) 
            throws ServletException, IOException {
        
        // Get or generate request ID
        String requestId = request.getHeader(REQUEST_ID_HEADER);
        if (requestId == null || requestId.isEmpty()) {
            requestId = UUID.randomUUID().toString();
        }
        
        // Get or generate trace ID (for distributed tracing)
        String traceId = request.getHeader(TRACE_ID_HEADER);
        if (traceId == null || traceId.isEmpty()) {
            traceId = requestId; // Use request ID as trace ID for the root request
        }
        
        // Set request ID and trace ID in response headers
        response.setHeader(REQUEST_ID_HEADER, requestId);
        response.setHeader(TRACE_ID_HEADER, traceId);
        
        // Store in MDC for logging
        org.slf4j.MDC.put("requestId", requestId);
        org.slf4j.MDC.put("traceId", traceId);
        
        try {
            // Continue the filter chain
            filterChain.doFilter(request, response);
        } finally {
            // Clean up MDC
            org.slf4j.MDC.remove("requestId");
            org.slf4j.MDC.remove("traceId");
        }
    }
}
