package com.metawebthree.common.metrics;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * REST controller for querying performance metrics
 */
@RestController
@RequestMapping("/metrics/performance")
public class PerformanceMetricsController {
    
    @Autowired
    private PerformanceMetricsService metricsService;
    
    /**
     * Get average execution time for a service/method
     */
    @GetMapping("/avg-time")
    public ResponseEntity<?> getAverageExecutionTime(
            @RequestParam("service") String serviceName,
            @RequestParam("method") String methodName) {
        
        double avgTime = metricsService.getAverageExecutionTime(serviceName, methodName);
        
        Map<String, Object> result = new HashMap<>();
        result.put("service", serviceName);
        result.put("method", methodName);
        result.put("averageExecutionTimeMs", avgTime);
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * Get error rate for a service/method
     */
    @GetMapping("/error-rate")
    public ResponseEntity<?> getErrorRate(
            @RequestParam("service") String serviceName,
            @RequestParam("method") String methodName) {
        
        double errorRate = metricsService.getErrorRate(serviceName, methodName);
        
        Map<String, Object> result = new HashMap<>();
        result.put("service", serviceName);
        result.put("method", methodName);
        result.put("errorRatePercent", errorRate);
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * Get overall performance summary for a service
     */
    @GetMapping("/summary")
    public ResponseEntity<?> getPerformanceSummary(
            @RequestParam("service") String serviceName) {
        
        Map<String, Object> summary = new HashMap<>();
        summary.put("service", serviceName);
        summary.put("averageExecutionTimeMs", 0.0); // Would need to aggregate all methods
        summary.put("errorRatePercent", 0.0); // Would need to aggregate all methods
        summary.put("status", "Metrics collection in progress");
        
        return ResponseEntity.ok(summary);
    }
}
