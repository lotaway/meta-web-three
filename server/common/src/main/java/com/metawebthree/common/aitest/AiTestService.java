package com.metawebthree.common.aitest;

import com.metawebthree.common.alert.AlertService;
import com.metawebthree.common.metrics.PerformanceMetricsService;
import com.metawebthree.common.trace.BottleneckAnalysis;
import com.metawebthree.common.trace.DistributedTraceService;
import com.metawebthree.common.trace.RootCauseAnalysis;
import com.metawebthree.common.trace.TraceRecord;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * AI-assisted testing and optimization service
 * 
 * Features:
 * - Intelligent performance test scenario generation based on service topology
 * - AI root cause diagnosis based on trace/metrics/log data
 * - Optimization suggestions based on historical data patterns
 * - Natural language test flow generation
 */
@Service
public class AiTestService {
    
    @Autowired
    private DistributedTraceService traceService;
    
    @Autowired
    private PerformanceMetricsService metricsService;
    
    @Autowired
    private AlertService alertService;
    
    // Historical data for pattern detection
    private final Map<String, ServicePerformanceHistory> performanceHistory = new ConcurrentHashMap<>();
    private final AtomicLong scenarioCounter = new AtomicLong(0);
    private final AtomicLong suggestionCounter = new AtomicLong(0);
    private final AtomicLong diagnosisCounter = new AtomicLong(0);

    // Known service endpoints for test scenario generation
    private static final Map<String, List<ServiceEndpoint>> SERVICE_ENDPOINTS = new LinkedHashMap<>();
    
    static {
        SERVICE_ENDPOINTS.put("order-service", Arrays.asList(
            new ServiceEndpoint("order-service", "/order/create", "POST", "Create a new order"),
            new ServiceEndpoint("order-service", "/order/{id}", "GET", "Get order details"),
            new ServiceEndpoint("order-service", "/order/list", "GET", "List orders"),
            new ServiceEndpoint("order-service", "/order/{id}/close", "POST", "Close an order"),
            new ServiceEndpoint("order-service", "/order/{id}/pay", "POST", "Pay for an order")
        ));
        SERVICE_ENDPOINTS.put("product-service", Arrays.asList(
            new ServiceEndpoint("product-service", "/product/{id}", "GET", "Get product details"),
            new ServiceEndpoint("product-service", "/product/list", "GET", "List products"),
            new ServiceEndpoint("product-service", "/product/search", "GET", "Search products"),
            new ServiceEndpoint("product-service", "/product/create", "POST", "Create a product")
        ));
        SERVICE_ENDPOINTS.put("user-service", Arrays.asList(
            new ServiceEndpoint("user-service", "/user/{id}", "GET", "Get user info"),
            new ServiceEndpoint("user-service", "/user/register", "POST", "Register user"),
            new ServiceEndpoint("user-service", "/user/login", "POST", "User login"),
            new ServiceEndpoint("user-service", "/user/{id}/profile", "PUT", "Update profile")
        ));
        SERVICE_ENDPOINTS.put("payment-service", Arrays.asList(
            new ServiceEndpoint("payment-service", "/payment/create", "POST", "Create payment"),
            new ServiceEndpoint("payment-service", "/payment/{id}", "GET", "Get payment status"),
            new ServiceEndpoint("payment-service", "/payment/refund", "POST", "Refund payment")
        ));
        SERVICE_ENDPOINTS.put("inventory-service", Arrays.asList(
            new ServiceEndpoint("inventory-service", "/inventory/{productId}", "GET", "Get inventory"),
            new ServiceEndpoint("inventory-service", "/inventory/deduct", "POST", "Deduct inventory"),
            new ServiceEndpoint("inventory-service", "/inventory/restock", "POST", "Restock inventory")
        ));
    }

    /**
     * Generate performance test scenarios for a given service or all services
     */
    public List<TestScenario> generatePerformanceTestScenarios(String targetService, String category) {
        List<TestScenario> scenarios = new ArrayList<>();
        
        if (targetService == null || targetService.isEmpty() || "all".equalsIgnoreCase(targetService)) {
            for (String service : SERVICE_ENDPOINTS.keySet()) {
                scenarios.addAll(generateScenariosForService(service, category));
            }
        } else {
            scenarios.addAll(generateScenariosForService(targetService, category));
        }
        
        return scenarios;
    }
    
    private List<TestScenario> generateScenariosForService(String serviceName, String category) {
        List<TestScenario> scenarios = new ArrayList<>();
        List<ServiceEndpoint> endpoints = SERVICE_ENDPOINTS.get(serviceName);
        if (endpoints == null) return scenarios;
        
        // Get historical performance data for context
        ServicePerformanceHistory history = performanceHistory.get(serviceName);
        
        // Scenario 1: Basic CRUD performance test
        if (category == null || "PERFORMANCE".equals(category)) {
            scenarios.add(generateCrudPerformanceScenario(serviceName, endpoints, history));
        }
        
        // Scenario 2: Concurrent load test
        if (category == null || "STRESS".equals(category)) {
            scenarios.add(generateStressTestScenario(serviceName, endpoints, history));
        }
        
        // Scenario 3: Reliability test (error handling)
        if (category == null || "RELIABILITY".equals(category)) {
            scenarios.add(generateReliabilityScenario(serviceName, endpoints, history));
        }
        
        // Scenario 4: Cross-service chain test
        if (category == null || "FUNCTIONAL".equals(category)) {
            scenarios.add(generateCrossServiceScenario(serviceName, history));
        }
        
        return scenarios;
    }
    
    private TestScenario generateCrudPerformanceScenario(String serviceName, List<ServiceEndpoint> endpoints, ServicePerformanceHistory history) {
        TestScenario scenario = new TestScenario();
        scenario.setScenarioId("TS-" + scenarioCounter.incrementAndGet());
        scenario.setName(serviceName + " CRUD Performance Test");
        scenario.setDescription("Performance test for " + serviceName + " basic CRUD operations");
        scenario.setCategory("PERFORMANCE");
        scenario.setTargetService(serviceName);
        scenario.setPriority(determinePriority(history));
        
        int stepOrder = 1;
        for (ServiceEndpoint endpoint : endpoints) {
            TestStep step = new TestStep();
            step.setOrder(stepOrder++);
            step.setAction(endpoint.description);
            step.setEndpoint(endpoint.path);
            step.setMethod(endpoint.method);
            step.setExpectedStatus("200");
            step.setExpectedMaxDurationMs(history != null ? Math.max(history.avgResponseMs * 3, 1000) : 1000);
            scenario.getSteps().add(step);
        }
        
        PerformanceExpectation expectation = new PerformanceExpectation();
        expectation.setMaxResponseTimeMs(history != null ? Math.max(history.maxResponseMs * 2, 3000) : 3000);
        expectation.setAvgResponseTimeMs(history != null ? Math.max(history.avgResponseMs * 2, 500) : 500);
        expectation.setMaxErrorRatePercent(1.0);
        expectation.setMinThroughputRps(100);
        expectation.setMaxP99LatencyMs(history != null ? Math.max(history.avgResponseMs * 5, 2000) : 2000);
        scenario.setPerformanceExpectation(expectation);
        
        return scenario;
    }
    
    private TestScenario generateStressTestScenario(String serviceName, List<ServiceEndpoint> endpoints, ServicePerformanceHistory history) {
        TestScenario scenario = new TestScenario();
        scenario.setScenarioId("TS-" + scenarioCounter.incrementAndGet());
        scenario.setName(serviceName + " Stress Test");
        scenario.setDescription("Stress test for " + serviceName + " under high concurrent load");
        scenario.setCategory("STRESS");
        scenario.setTargetService(serviceName);
        scenario.setPriority("HIGH");
        
        // Pick the most critical endpoint (POST/PUT for writes, or first GET)
        ServiceEndpoint criticalEndpoint = endpoints.stream()
                .filter(e -> "POST".equals(e.method))
                .findFirst()
                .orElse(endpoints.get(0));
        
        TestStep step = new TestStep();
        step.setOrder(1);
        step.setAction("Concurrent load test: " + criticalEndpoint.description);
        step.setEndpoint(criticalEndpoint.path);
        step.setMethod(criticalEndpoint.method);
        step.setExpectedStatus("200");
        step.setExpectedMaxDurationMs(5000);
        scenario.getSteps().add(step);
        
        scenario.getPrerequisites().add("Ensure " + serviceName + " is running and healthy");
        scenario.getPrerequisites().add("Configure load generator for 100-500 concurrent users");
        scenario.getPrerequisites().add("Monitor system resources during test");
        
        PerformanceExpectation expectation = new PerformanceExpectation();
        expectation.setMaxResponseTimeMs(5000);
        expectation.setAvgResponseTimeMs(1000);
        expectation.setMaxErrorRatePercent(5.0);
        expectation.setMinThroughputRps(50);
        expectation.setMaxP99LatencyMs(3000);
        scenario.setPerformanceExpectation(expectation);
        
        return scenario;
    }
    
    private TestScenario generateReliabilityScenario(String serviceName, List<ServiceEndpoint> endpoints, ServicePerformanceHistory history) {
        TestScenario scenario = new TestScenario();
        scenario.setScenarioId("TS-" + scenarioCounter.incrementAndGet());
        scenario.setName(serviceName + " Reliability Test");
        scenario.setDescription("Test " + serviceName + " error handling and recovery");
        scenario.setCategory("RELIABILITY");
        scenario.setTargetService(serviceName);
        scenario.setPriority("HIGH");
        
        TestStep invalidInput = new TestStep();
        invalidInput.setOrder(1);
        invalidInput.setAction("Send invalid input to " + serviceName);
        invalidInput.setEndpoint(endpoints.get(0).path);
        invalidInput.setMethod(endpoints.get(0).method);
        invalidInput.setRequestBody("{\"invalid\": \"data\"}");
        invalidInput.setExpectedStatus("400");
        invalidInput.setExpectedMaxDurationMs(500);
        scenario.getSteps().add(invalidInput);
        
        TestStep validRequest = new TestStep();
        validRequest.setOrder(2);
        validRequest.setAction("Verify service recovery after error");
        validRequest.setEndpoint(endpoints.get(0).path);
        validRequest.setMethod("GET");
        validRequest.setExpectedStatus("200");
        validRequest.setExpectedMaxDurationMs(1000);
        scenario.getSteps().add(validRequest);
        
        PerformanceExpectation expectation = new PerformanceExpectation();
        expectation.setMaxResponseTimeMs(1000);
        expectation.setMaxErrorRatePercent(0.0);
        scenario.setPerformanceExpectation(expectation);
        
        return scenario;
    }
    
    private TestScenario generateCrossServiceScenario(String serviceName, ServicePerformanceHistory history) {
        TestScenario scenario = new TestScenario();
        scenario.setScenarioId("TS-" + scenarioCounter.incrementAndGet());
        scenario.setName(serviceName + " Cross-Service Chain Test");
        scenario.setDescription("Test " + serviceName + " in a typical cross-service business flow");
        scenario.setCategory("FUNCTIONAL");
        scenario.setTargetService(serviceName);
        scenario.setPriority("MEDIUM");
        
        // Generate a typical e-commerce flow
        if ("order-service".equals(serviceName)) {
            scenario.getSteps().add(createStep(1, "Query product inventory", "/inventory/{productId}", "GET", 1000));
            scenario.getSteps().add(createStep(2, "Get product details", "/product/{id}", "GET", 800));
            scenario.getSteps().add(createStep(3, "Create order", "/order/create", "POST", 2000));
            scenario.getSteps().add(createStep(4, "Process payment", "/payment/create", "POST", 3000));
            scenario.getSteps().add(createStep(5, "Deduct inventory", "/inventory/deduct", "POST", 1000));
        } else {
            scenario.getSteps().add(createStep(1, "Call " + serviceName + " primary endpoint", 
                    SERVICE_ENDPOINTS.getOrDefault(serviceName, Collections.emptyList())
                            .stream().findFirst().map(e -> e.path).orElse("/" + serviceName + "/list"),
                    "GET", 1500));
        }
        
        PerformanceExpectation expectation = new PerformanceExpectation();
        expectation.setMaxResponseTimeMs(5000);
        expectation.setAvgResponseTimeMs(1500);
        expectation.setMaxErrorRatePercent(1.0);
        scenario.setPerformanceExpectation(expectation);
        
        return scenario;
    }
    
    private TestStep createStep(int order, String action, String endpoint, String method, long maxDurationMs) {
        TestStep step = new TestStep();
        step.setOrder(order);
        step.setAction(action);
        step.setEndpoint(endpoint);
        step.setMethod(method);
        step.setExpectedStatus("200");
        step.setExpectedMaxDurationMs(maxDurationMs);
        return step;
    }
    
    private String determinePriority(ServicePerformanceHistory history) {
        if (history == null) return "MEDIUM";
        if (history.errorRate > 5.0 || history.avgResponseMs > 2000) return "HIGH";
        if (history.errorRate > 1.0 || history.avgResponseMs > 1000) return "MEDIUM";
        return "LOW";
    }
    
    /**
     * Perform AI root cause diagnosis for a given trace
     * Analyzes trace data, correlates with metrics and logs
     */
    public AiDiagnosisResult performAiDiagnosis(String traceId) {
        AiDiagnosisResult result = new AiDiagnosisResult();
        result.setDiagnosisId("DG-" + diagnosisCounter.incrementAndGet());
        result.setTraceId(traceId);
        
        // Get trace records and root cause analysis
        List<TraceRecord> records = traceService.getTraceRecords(traceId);
        RootCauseAnalysis rootCause = traceService.performRootCauseAnalysis(traceId);
        BottleneckAnalysis bottleneck = traceService.analyzeBottlenecks(traceId);
        
        if (records.isEmpty()) {
            result.setDiagnosisSuccessful(false);
            result.setRootCauseSummary("No trace data available for diagnosis");
            return result;
        }
        
        result.setDiagnosisSuccessful(true);
        result.setAffectedService(rootCause.getRootCauseService());
        result.setAffectedMethod(rootCause.getRootCauseMethod());
        
        // Classify root cause type based on patterns
        String rootCauseType = classifyRootCause(records, rootCause);
        result.setRootCauseType(rootCauseType);
        
        // Build summary
        if (rootCause.isRootCauseFound()) {
            result.setRootCauseSummary(String.format(
                "Root cause: %s in %s.%s — %s (impact: %.1f%%)",
                rootCause.getRootCauseType(),
                rootCause.getRootCauseService(),
                rootCause.getRootCauseMethod(),
                rootCause.getRootCauseDetail(),
                rootCause.getImpactPercentage()
            ));
        } else {
            result.setRootCauseSummary("No significant performance issues found in this trace");
        }
        
        // Build evidence chain
        List<String> evidence = new ArrayList<>();
        evidence.add(String.format("Trace contains %d spans, total duration: %dms",
                records.size(), bottleneck.getTotalDurationMs()));
        
        if (bottleneck.isBottleneckDetected()) {
            evidence.add(String.format("Bottleneck: %s.%s (%.1f%% of total time)",
                    bottleneck.getBottleneckService(), bottleneck.getBottleneckMethod(),
                    bottleneck.getBottleneckPercentage()));
        }
        
        long errorCount = records.stream().filter(r -> "ERROR".equals(r.getStatus())).count();
        if (errorCount > 0) {
            evidence.add(String.format("%d error spans detected in trace", errorCount));
        }
        
        long slowCount = records.stream().filter(r -> r.getDurationMs() != null && r.getDurationMs() > 3000).count();
        if (slowCount > 0) {
            evidence.add(String.format("%d slow spans (>3s) detected", slowCount));
        }
        result.setEvidenceChain(evidence);
        
        // Generate remediation steps based on root cause type
        result.setRemediationSteps(generateRemediationSteps(rootCauseType, rootCause));
        
        // Find similar historical incidents
        result.setSimilarHistoricalIncidents(findSimilarIncidents(rootCause));
        
        // Calculate confidence score
        result.setConfidenceScore(calculateConfidenceScore(records, rootCause));
        
        return result;
    }
    
    private String classifyRootCause(List<TraceRecord> records, RootCauseAnalysis rootCause) {
        if (!rootCause.isRootCauseFound()) return "UNKNOWN";
        
        // Check for cascade failure patterns
        long errorCount = records.stream().filter(r -> "ERROR".equals(r.getStatus())).count();
        if (errorCount > 2) return "CASCADE_FAILURE";
        
        // Check for slow query patterns (single span dominates)
        if (rootCause.getImpactPercentage() > 60) return "SLOW_QUERY";
        
        // Check for resource exhaustion (multiple slow spans)
        long slowCount = records.stream().filter(r -> r.getDurationMs() != null && r.getDurationMs() > 3000).count();
        if (slowCount > 2) return "RESOURCE_EXHAUSTION";
        
        // Default to the root cause type from analysis
        return "ERROR".equals(rootCause.getRootCauseType()) ? "CONFIGURATION_ERROR" : "NETWORK_ISSUE";
    }
    
    private List<String> generateRemediationSteps(String rootCauseType, RootCauseAnalysis rootCause) {
        List<String> steps = new ArrayList<>();
        
        switch (rootCauseType) {
            case "SLOW_QUERY":
                steps.add("1. Analyze the slow method's database queries for missing indexes");
                steps.add("2. Enable query logging and review execution plans");
                steps.add("3. Consider adding caching (Redis) for frequently accessed data");
                steps.add("4. Review if batch processing can replace individual queries");
                break;
            case "CASCADE_FAILURE":
                steps.add("1. Add circuit breakers (Resilience4j) to prevent cascade failures");
                steps.add("2. Implement timeout policies for inter-service calls");
                steps.add("3. Add fallback methods for critical service dependencies");
                steps.add("4. Review and adjust retry policies to avoid retry storms");
                break;
            case "RESOURCE_EXHAUSTION":
                steps.add("1. Check service resource usage (CPU, memory, connections)");
                steps.add("2. Review thread pool and connection pool configurations");
                steps.add("3. Consider horizontal scaling for the affected service");
                steps.add("4. Implement rate limiting to protect against traffic spikes");
                break;
            case "CONFIGURATION_ERROR":
                steps.add("1. Review service configuration and environment variables");
                steps.add("2. Check database connection strings and credentials");
                steps.add("3. Verify service discovery registration");
                steps.add("4. Review recent configuration changes");
                break;
            case "NETWORK_ISSUE":
                steps.add("1. Check network connectivity between services");
                steps.add("2. Review DNS resolution and load balancer health");
                steps.add("3. Check for network latency between service nodes");
                steps.add("4. Verify firewall and security group settings");
                break;
            default:
                steps.add("1. Collect more diagnostic data (logs, metrics, traces)");
                steps.add("2. Monitor the service for recurring patterns");
        }
        
        return steps;
    }
    
    private List<String> findSimilarIncidents(RootCauseAnalysis rootCause) {
        List<String> incidents = new ArrayList<>();
        
        // Check performance history for similar patterns
        for (Map.Entry<String, ServicePerformanceHistory> entry : performanceHistory.entrySet()) {
            ServicePerformanceHistory hist = entry.getValue();
            if (hist.errorRate > 5.0) {
                incidents.add(String.format("Service %s has high error rate (%.1f%%)", 
                        entry.getKey(), hist.errorRate));
            }
            if (hist.avgResponseMs > 2000) {
                incidents.add(String.format("Service %s has slow avg response (%dms)", 
                        entry.getKey(), hist.avgResponseMs));
            }
        }
        
        return incidents;
    }
    
    private double calculateConfidenceScore(List<TraceRecord> records, RootCauseAnalysis rootCause) {
        if (!rootCause.isRootCauseFound()) return 0.3;
        
        double score = 0.5;
        
        // More spans = more data = higher confidence
        score += Math.min(records.size() * 0.05, 0.2);
        
        // Error evidence increases confidence
        long errorCount = records.stream().filter(r -> "ERROR".equals(r.getStatus())).count();
        score += Math.min(errorCount * 0.1, 0.15);
        
        // Higher impact percentage increases confidence
        if (rootCause.getImpactPercentage() > 50) score += 0.15;
        
        return Math.min(score, 1.0);
    }
    
    /**
     * Generate optimization suggestions based on historical performance data
     */
    public List<OptimizationSuggestion> generateOptimizationSuggestions(String targetService) {
        List<OptimizationSuggestion> suggestions = new ArrayList<>();
        
        Collection<ServicePerformanceHistory> histories = targetService != null && !targetService.isEmpty()
                ? Collections.singletonList(performanceHistory.get(targetService))
                : performanceHistory.values();
        
        for (ServicePerformanceHistory history : histories) {
            if (history == null) continue;
            
            // High response time suggestion
            if (history.avgResponseMs > 1000) {
                suggestions.add(createSuggestion(
                    history.serviceName, "PERFORMANCE", "HIGH",
                    "Optimize " + history.serviceName + " response time",
                    String.format("%s avg response time is %dms, exceeding 1s threshold",
                            history.serviceName, history.avgResponseMs),
                    "Current avg: " + history.avgResponseMs + "ms",
                    "Add caching, optimize queries, or scale horizontally",
                    "Expected 30-50% response time reduction",
                    history.serviceName, null, 0.8
                ));
            }
            
            // High error rate suggestion
            if (history.errorRate > 2.0) {
                suggestions.add(createSuggestion(
                    history.serviceName, "RELIABILITY", "CRITICAL",
                    "Reduce " + history.serviceName + " error rate",
                    String.format("%s error rate is %.1f%%, exceeding 2%% threshold",
                            history.serviceName, history.errorRate),
                    "Current error rate: " + String.format("%.1f%%", history.errorRate),
                    "Add error handling, retry logic, and circuit breakers",
                    "Expected error rate reduction to <1%",
                    history.serviceName, null, 0.9
                ));
            }
            
            // Max response time too high
            if (history.maxResponseMs > 5000) {
                suggestions.add(createSuggestion(
                    history.serviceName, "PERFORMANCE", "MEDIUM",
                    "Investigate " + history.serviceName + " worst-case latency",
                    String.format("%s max response time is %dms (>5s)",
                            history.serviceName, history.maxResponseMs),
                    "Max response: " + history.maxResponseMs + "ms",
                    "Profile slow methods, add timeout policies",
                    "Expected max response reduction to <3s",
                    history.serviceName, null, 0.7
                ));
            }
        }
        
        // Add general best-practice suggestions if no specific issues found
        if (suggestions.isEmpty()) {
            suggestions.add(createSuggestion(
                targetService != null ? targetService : "all", "SCALABILITY", "LOW",
                "Consider implementing auto-scaling policies",
                "No critical performance issues detected, but proactive scaling recommended",
                "Current: manual scaling",
                "Implement HPA/VPA based on CPU/memory/custom metrics",
                "Better resource utilization and cost optimization",
                targetService, null, 0.5
            ));
        }
        
        return suggestions;
    }
    
    private OptimizationSuggestion createSuggestion(String id, String category, String severity,
            String title, String description, String currentBehavior, String suggestedChange,
            String expectedImprovement, String targetService, String targetMethod, double confidence) {
        OptimizationSuggestion s = new OptimizationSuggestion();
        s.setSuggestionId("OS-" + suggestionCounter.incrementAndGet());
        s.setCategory(category);
        s.setSeverity(severity);
        s.setTitle(title);
        s.setDescription(description);
        s.setCurrentBehavior(currentBehavior);
        s.setSuggestedChange(suggestedChange);
        s.setExpectedImprovement(expectedImprovement);
        s.setTargetService(targetService);
        s.setTargetMethod(targetMethod);
        s.setConfidenceScore(confidence);
        return s;
    }
    
    /**
     * Generate a natural language test flow description
     * Converts technical test scenarios into readable test procedures
     */
    public String generateNaturalLanguageTestFlow(TestScenario scenario) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("# Test Flow: ").append(scenario.getName()).append("\n\n");
        sb.append("## Overview\n");
        sb.append(scenario.getDescription()).append("\n\n");
        sb.append("**Category:** ").append(scenario.getCategory()).append("\n");
        sb.append("**Priority:** ").append(scenario.getPriority()).append("\n");
        sb.append("**Target Service:** ").append(scenario.getTargetService()).append("\n\n");
        
        if (!scenario.getPrerequisites().isEmpty()) {
            sb.append("## Prerequisites\n");
            for (String prereq : scenario.getPrerequisites()) {
                sb.append("- ").append(prereq).append("\n");
            }
            sb.append("\n");
        }
        
        sb.append("## Test Steps\n\n");
        for (TestStep step : scenario.getSteps()) {
            sb.append(String.format("### Step %d: %s\n", step.getOrder(), step.getAction()));
            sb.append(String.format("- Send a **%s** request to `%s`\n", step.getMethod(), step.getEndpoint()));
            if (step.getRequestBody() != null) {
                sb.append(String.format("- Request body: `%s`\n", step.getRequestBody()));
            }
            sb.append(String.format("- Expected HTTP status: **%s**\n", step.getExpectedStatus()));
            sb.append(String.format("- Maximum allowed duration: **%dms**\n\n", step.getExpectedMaxDurationMs()));
        }
        
        if (scenario.getPerformanceExpectation() != null) {
            PerformanceExpectation pe = scenario.getPerformanceExpectation();
            sb.append("## Performance Expectations\n");
            sb.append(String.format("- Maximum response time: **%dms**\n", pe.getMaxResponseTimeMs()));
            sb.append(String.format("- Average response time: **%dms**\n", pe.getAvgResponseTimeMs()));
            sb.append(String.format("- Maximum error rate: **%.1f%%**\n", pe.getMaxErrorRatePercent()));
            sb.append(String.format("- Minimum throughput: **%d req/s**\n", pe.getMinThroughputRps()));
            if (pe.getMaxP99LatencyMs() > 0) {
                sb.append(String.format("- P99 latency threshold: **%dms**\n", pe.getMaxP99LatencyMs()));
            }
        }
        
        return sb.toString();
    }
    
    /**
     * Record performance data for a service (called by PerformanceMetricsAspect)
     * Used to build historical data for optimization suggestions
     */
    public void recordServicePerformance(String serviceName, long durationMs, boolean isError) {
        ServicePerformanceHistory history = performanceHistory.computeIfAbsent(
                serviceName, ServicePerformanceHistory::new);
        history.record(durationMs, isError);
    }
    
    /**
     * Get performance history for a service
     */
    public ServicePerformanceHistory getPerformanceHistory(String serviceName) {
        return performanceHistory.get(serviceName);
    }
    
    /**
     * Internal class for tracking service performance history
     */
    public static class ServicePerformanceHistory {
        public final String serviceName;
        public long totalCalls;
        public long totalDurationMs;
        public long maxResponseMs;
        public long avgResponseMs;
        public long errorCount;
        public double errorRate;
        
        public ServicePerformanceHistory(String serviceName) {
            this.serviceName = serviceName;
        }
        
        public void record(long durationMs, boolean isError) {
            totalCalls++;
            totalDurationMs += durationMs;
            if (durationMs > maxResponseMs) {
                maxResponseMs = durationMs;
            }
            avgResponseMs = totalDurationMs / totalCalls;
            if (isError) errorCount++;
            errorRate = totalCalls > 0 ? (double) errorCount / totalCalls * 100.0 : 0.0;
        }
    }
    
    /**
     * Internal class representing a known service endpoint
     */
    private static class ServiceEndpoint {
        final String serviceName;
        final String path;
        final String method;
        final String description;
        
        ServiceEndpoint(String serviceName, String path, String method, String description) {
            this.serviceName = serviceName;
            this.path = path;
            this.method = method;
            this.description = description;
        }
    }
}
