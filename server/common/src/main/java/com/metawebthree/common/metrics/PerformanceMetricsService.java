package com.metawebthree.common.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.Counter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

/**
 * Service for collecting and monitoring service performance metrics
 * Tracks response time, throughput, and error rates
 */
@Service
public class PerformanceMetricsService {
    
    @Autowired
    private MeterRegistry meterRegistry;
    
    private static final String METRIC_PREFIX = "metawebthree.service.";
    
    /**
     * Record method execution time
     */
    public void recordExecutionTime(String serviceName, String methodName, long durationMs) {
        String metricName = METRIC_PREFIX + "execution.time";
        Timer timer = Timer.builder(metricName)
                .tag("service", serviceName)
                .tag("method", methodName)
                .description("Method execution time in milliseconds")
                .register(meterRegistry);
        timer.record(durationMs, TimeUnit.MILLISECONDS);
    }
    
    /**
     * Record successful request (for throughput tracking)
     */
    public void recordSuccess(String serviceName, String methodName) {
        String metricName = METRIC_PREFIX + "requests.success";
        Counter counter = Counter.builder(metricName)
                .tag("service", serviceName)
                .tag("method", methodName)
                .description("Successful request count")
                .register(meterRegistry);
        counter.increment();
    }
    
    /**
     * Record failed request (for error rate tracking)
     */
    public void recordError(String serviceName, String methodName, String errorType) {
        String metricName = METRIC_PREFIX + "requests.error";
        Counter counter = Counter.builder(metricName)
                .tag("service", serviceName)
                .tag("method", methodName)
                .tag("errorType", errorType)
                .description("Failed request count")
                .register(meterRegistry);
        counter.increment();
    }
    
    /**
     * Record request count (for throughput)
     */
    public void recordRequest(String serviceName, String methodName) {
        String metricName = METRIC_PREFIX + "requests.total";
        Counter counter = Counter.builder(metricName)
                .tag("service", serviceName)
                .tag("method", methodName)
                .description("Total request count")
                .register(meterRegistry);
        counter.increment();
    }
    
    /**
     * Get average execution time for a service/method
     */
    public double getAverageExecutionTime(String serviceName, String methodName) {
        String metricName = METRIC_PREFIX + "execution.time";
        return meterRegistry.get(metricName)
                .tags("service", serviceName, "method", methodName)
                .timer()
                .mean(TimeUnit.MILLISECONDS);
    }
    
    /**
     * Get error rate for a service/method
     */
    public double getErrorRate(String serviceName, String methodName) {
        String successMetric = METRIC_PREFIX + "requests.success";
        String errorMetric = METRIC_PREFIX + "requests.error";
        String totalMetric = METRIC_PREFIX + "requests.total";
        
        double successCount = meterRegistry.get(successMetric)
                .tags("service", serviceName, "method", methodName)
                .counter()
                .count();
        
        double errorCount = meterRegistry.get(errorMetric)
                .tags("service", serviceName, "method", methodName, "errorType", "all")
                .counter()
                .count();
        
        double totalCount = meterRegistry.get(totalMetric)
                .tags("service", serviceName, "method", methodName)
                .counter()
                .count();
        
        if (totalCount == 0) {
            return 0.0;
        }
        
        return errorCount / totalCount * 100.0;
    }
}
