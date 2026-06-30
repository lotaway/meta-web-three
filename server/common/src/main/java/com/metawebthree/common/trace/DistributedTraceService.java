package com.metawebthree.common.trace;

import org.apache.skywalking.apm.toolkit.trace.TraceContext;
import org.slf4j.MDC;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Service for distributed tracing across microservices
 * Integrates with SkyWalking for full-chain performance monitoring
 * 
 * Features:
 * - Distributed trace span management with request ID propagation
 * - Call chain visualization via trace records
 * - Service topology auto-discovery from trace data
 * - Performance bottleneck detection and root cause analysis
 */
@Service
public class DistributedTraceService {
    
    private final Map<String, TraceRecord> traceRecords = new ConcurrentHashMap<>();
    
    // Service call statistics for topology building
    // Key: "sourceService->targetService"
    private final Map<String, ServiceCallStats> serviceCallStats = new ConcurrentHashMap<>();
    
    // Per-service statistics
    // Key: serviceName
    private final Map<String, ServiceStats> serviceStats = new ConcurrentHashMap<>();

    // Track parent-child span relationships for call chain reconstruction
    // Key: traceId, Value: ordered list of span IDs
    private final Map<String, List<String>> traceSpanOrder = new ConcurrentHashMap<>();
    
    /**
     * Start a new trace span
     */
    public TraceSpan startSpan(String serviceName, String methodName) {
        return startSpan(serviceName, methodName, null);
    }
    
    /**
     * Start a new trace span with caller info for dependency tracking
     */
    public TraceSpan startSpan(String serviceName, String methodName, String callerService) {
        String traceId = getTraceId();
        String spanId = generateSpanId();
        
        TraceSpan span = new TraceSpan();
        span.setTraceId(traceId);
        span.setSpanId(spanId);
        span.setServiceName(serviceName);
        span.setMethodName(methodName);
        span.setStartTime(System.currentTimeMillis());
        span.setStatus("STARTED");
        
        // Store the span
        String key = traceId + ":" + spanId;
        TraceRecord record = new TraceRecord();
        record.setTraceId(traceId);
        record.setSpanId(spanId);
        record.setServiceName(serviceName);
        record.setMethodName(methodName);
        record.setStartTime(span.getStartTime());
        record.setStatus("STARTED");
        
        traceRecords.put(key, record);
        
        // Track span order for this trace
        traceSpanOrder.computeIfAbsent(traceId, k -> new ArrayList<>()).add(spanId);
        
        // Record service call dependency
        if (callerService != null && !callerService.equals(serviceName)) {
            String depKey = callerService + "->" + serviceName;
            serviceCallStats.computeIfAbsent(depKey, k -> new ServiceCallStats(callerService, serviceName))
                    .incrementCalls();
        }
        
        // Update per-service stats
        serviceStats.computeIfAbsent(serviceName, k -> new ServiceStats(serviceName))
                .incrementCalls();
        
        // Log with trace ID
        org.slf4j.LoggerFactory.getLogger(DistributedTraceService.class)
            .info("Started span: {}.{} [traceId={}, spanId={}]", 
                  serviceName, methodName, traceId, spanId);
        
        return span;
    }
    
    /**
     * End a trace span and record the duration
     */
    public void endSpan(TraceSpan span, String status, String errorMessage) {
        long endTime = System.currentTimeMillis();
        long duration = endTime - span.getStartTime();
        
        span.setEndTime(endTime);
        span.setDurationMs(duration);
        span.setStatus(status);
        span.setErrorMessage(errorMessage);
        
        // Update the trace record
        String key = span.getTraceId() + ":" + span.getSpanId();
        TraceRecord record = traceRecords.get(key);
        if (record != null) {
            record.setEndTime(endTime);
            record.setDurationMs(duration);
            record.setStatus(status);
            record.setErrorMessage(errorMessage);
        }
        
        // Update service stats
        ServiceStats stats = serviceStats.get(span.getServiceName());
        if (stats != null) {
            stats.recordDuration(duration);
            if ("ERROR".equals(status)) {
                stats.incrementErrors();
            }
        }
        
        // Update service call stats
        // Find the caller from the trace chain
        updateServiceCallDuration(span);
        
        // Log with trace ID
        org.slf4j.LoggerFactory.getLogger(DistributedTraceService.class)
            .info("Ended span: {}.{} [traceId={}, spanId={}, duration={}ms, status={}]", 
                  span.getServiceName(), span.getMethodName(), 
                  span.getTraceId(), span.getSpanId(), duration, status);
        
        // Alert if duration is too long (> 3 seconds)
        if (duration > 3000) {
            org.slf4j.LoggerFactory.getLogger(DistributedTraceService.class)
                .warn("Slow span detected: {}.{} [traceId={}, duration={}ms]", 
                      span.getServiceName(), span.getMethodName(), 
                      span.getTraceId(), duration);
        }
    }
    
    /**
     * Update service call duration stats from span relationships
     */
    private void updateServiceCallDuration(TraceSpan span) {
        List<String> spanOrder = traceSpanOrder.get(span.getTraceId());
        if (spanOrder == null) return;
        
        int idx = spanOrder.indexOf(span.getSpanId());
        if (idx <= 0) return;
        
        // The previous span's service is likely the caller
        String prevSpanId = spanOrder.get(idx - 1);
        TraceRecord prevRecord = traceRecords.get(span.getTraceId() + ":" + prevSpanId);
        if (prevRecord != null && !prevRecord.getServiceName().equals(span.getServiceName())) {
            String depKey = prevRecord.getServiceName() + "->" + span.getServiceName();
            ServiceCallStats callStats = serviceCallStats.get(depKey);
            if (callStats != null) {
                callStats.recordDuration(span.getDurationMs() != null ? span.getDurationMs() : 0);
                if ("ERROR".equals(span.getStatus())) {
                    callStats.incrementErrors();
                }
            }
        }
    }
    
    /**
     * Get the current trace ID (from SkyWalking or MDC)
     */
    public String getTraceId() {
        // Try to get from SkyWalking first
        try {
            String skywalkingTraceId = TraceContext.traceId();
            if (skywalkingTraceId != null && !skywalkingTraceId.isEmpty()) {
                return skywalkingTraceId;
            }
        } catch (Exception ignored) {
            // SkyWalking agent may not be present
        }
        
        // Fall back to MDC
        String mdcTraceId = MDC.get("traceId");
        if (mdcTraceId != null && !mdcTraceId.isEmpty()) {
            return mdcTraceId;
        }
        
        // Generate new trace ID
        return "trace-" + System.currentTimeMillis() + "-" + (int)(Math.random() * 10000);
    }
    
    /**
     * Generate a new span ID
     */
    private String generateSpanId() {
        return "span-" + System.currentTimeMillis() + "-" + (int)(Math.random() * 10000);
    }
    
    /**
     * Get all trace records for a given trace ID
     */
    public List<TraceRecord> getTraceRecords(String traceId) {
        List<TraceRecord> records = new ArrayList<>();
        traceRecords.forEach((key, record) -> {
            if (record.getTraceId().equals(traceId)) {
                records.add(record);
            }
        });
        // Sort by start time for call chain visualization
        records.sort(Comparator.comparingLong(TraceRecord::getStartTime));
        return records;
    }
    
    /**
     * Get all trace IDs (for listing recent traces)
     */
    public List<String> getRecentTraceIds(int limit) {
        return traceRecords.values().stream()
                .map(TraceRecord::getTraceId)
                .distinct()
                .limit(limit)
                .collect(Collectors.toList());
    }
    
    /**
     * Get slow traces (traces containing spans slower than the threshold)
     */
    public List<TraceRecord> getSlowSpans(long minDurationMs) {
        return traceRecords.values().stream()
                .filter(r -> r.getDurationMs() != null && r.getDurationMs() > minDurationMs)
                .sorted(Comparator.comparingLong(TraceRecord::getDurationMs).reversed())
                .collect(Collectors.toList());
    }
    
    /**
     * Analyze trace for performance bottlenecks
     */
    public BottleneckAnalysis analyzeBottlenecks(String traceId) {
        List<TraceRecord> records = getTraceRecords(traceId);
        BottleneckAnalysis analysis = new BottleneckAnalysis();
        analysis.setTraceId(traceId);
        
        TraceRecord slowestSpan = null;
        long totalDuration = 0;
        
        for (TraceRecord record : records) {
            if (record.getDurationMs() != null) {
                totalDuration += record.getDurationMs();
                
                if (slowestSpan == null || 
                    record.getDurationMs() > slowestSpan.getDurationMs()) {
                    slowestSpan = record;
                }
            }
        }
        
        analysis.setTotalDurationMs(totalDuration);
        analysis.setSlowestSpan(slowestSpan);
        analysis.setSpanCount(records.size());
        
        // Identify bottlenecks (spans taking more than 30% of total time)
        if (slowestSpan != null && totalDuration > 0) {
            double percentage = (double) slowestSpan.getDurationMs() / totalDuration * 100;
            if (percentage > 30) {
                analysis.setBottleneckDetected(true);
                analysis.setBottleneckService(slowestSpan.getServiceName());
                analysis.setBottleneckMethod(slowestSpan.getMethodName());
                analysis.setBottleneckPercentage(percentage);
            }
        }
        
        return analysis;
    }
    
    /**
     * Build service dependency topology from collected trace data
     * Automatically identifies service dependencies from actual call chains
     */
    public ServiceTopology buildServiceTopology() {
        ServiceTopology topology = new ServiceTopology();
        topology.setGeneratedAt(System.currentTimeMillis());
        
        // Build nodes from service stats
        Map<String, ServiceNode> nodeMap = new LinkedHashMap<>();
        for (Map.Entry<String, ServiceStats> entry : serviceStats.entrySet()) {
            ServiceStats stats = entry.getValue();
            ServiceNode node = new ServiceNode(stats.serviceName);
            node.setCallCount(stats.callCount);
            node.setTotalDurationMs(stats.totalDurationMs);
            node.setMaxDurationMs(stats.maxDurationMs);
            node.setAvgDurationMs(stats.callCount > 0 ? stats.totalDurationMs / stats.callCount : 0);
            node.setErrorCount(stats.errorCount);
            nodeMap.put(stats.serviceName, node);
        }
        
        // Build edges and update downstream services on nodes
        for (Map.Entry<String, ServiceCallStats> entry : serviceCallStats.entrySet()) {
            ServiceCallStats stats = entry.getValue();
            ServiceEdge edge = new ServiceEdge(stats.sourceService, stats.targetService);
            edge.setCallCount(stats.callCount);
            edge.setTotalDurationMs(stats.totalDurationMs);
            edge.setAvgDurationMs(stats.callCount > 0 ? stats.totalDurationMs / stats.callCount : 0);
            edge.setErrorCount(stats.errorCount);
            topology.addEdge(edge);
            
            // Update downstream list on source node
            ServiceNode sourceNode = nodeMap.get(stats.sourceService);
            if (sourceNode != null) {
                sourceNode.addDownstreamService(stats.targetService);
            }
            
            // Ensure target node exists
            if (!nodeMap.containsKey(stats.targetService)) {
                ServiceNode targetNode = new ServiceNode(stats.targetService);
                nodeMap.put(stats.targetService, targetNode);
            }
        }
        
        topology.setNodes(new ArrayList<>(nodeMap.values()));
        return topology;
    }
    
    /**
     * Perform root cause analysis on a trace
     * Identifies the most likely root cause of performance issues
     */
    public RootCauseAnalysis performRootCauseAnalysis(String traceId) {
        RootCauseAnalysis analysis = new RootCauseAnalysis();
        analysis.setTraceId(traceId);
        
        List<TraceRecord> records = getTraceRecords(traceId);
        if (records.isEmpty()) {
            analysis.setRootCauseFound(false);
            analysis.setRootCauseDetail("No trace records found for traceId: " + traceId);
            return analysis;
        }
        
        // Calculate total trace duration
        long minStart = records.stream().mapToLong(TraceRecord::getStartTime).min().orElse(0);
        long maxEnd = records.stream().mapToLong(r -> r.getEndTime() != null ? r.getEndTime() : r.getStartTime()).max().orElse(minStart);
        long totalTraceDuration = maxEnd - minStart;
        
        // Find error spans first (errors are more likely root causes)
        List<TraceRecord> errorSpans = records.stream()
                .filter(r -> "ERROR".equals(r.getStatus()))
                .collect(Collectors.toList());
        
        // Find slow spans (>30% of total duration)
        List<TraceRecord> slowSpans = records.stream()
                .filter(r -> r.getDurationMs() != null && totalTraceDuration > 0 
                        && (double) r.getDurationMs() / totalTraceDuration > 0.3)
                .sorted(Comparator.comparingLong(TraceRecord::getDurationMs).reversed())
                .collect(Collectors.toList());
        
        // Prioritize: errors > slow spans
        TraceRecord rootCauseRecord = null;
        String rootCauseType = null;
        
        if (!errorSpans.isEmpty()) {
            // First error in the chain is likely the root cause
            rootCauseRecord = errorSpans.get(0);
            rootCauseType = "ERROR";
        } else if (!slowSpans.isEmpty()) {
            rootCauseRecord = slowSpans.get(0);
            rootCauseType = "SLOW";
        }
        
        if (rootCauseRecord != null) {
            analysis.setRootCauseFound(true);
            analysis.setRootCauseService(rootCauseRecord.getServiceName());
            analysis.setRootCauseMethod(rootCauseRecord.getMethodName());
            analysis.setRootCauseType(rootCauseType);
            analysis.setRootCauseDurationMs(rootCauseRecord.getDurationMs() != null ? rootCauseRecord.getDurationMs() : 0);
            
            if (totalTraceDuration > 0) {
                analysis.setImpactPercentage(
                    (double) (rootCauseRecord.getDurationMs() != null ? rootCauseRecord.getDurationMs() : 0) 
                    / totalTraceDuration * 100.0
                );
            }
            
            if ("ERROR".equals(rootCauseType)) {
                analysis.setRootCauseDetail(String.format(
                    "Service %s.%s failed with error: %s. This error likely caused cascading failures downstream.",
                    rootCauseRecord.getServiceName(), 
                    rootCauseRecord.getMethodName(),
                    rootCauseRecord.getErrorMessage()
                ));
            } else {
                analysis.setRootCauseDetail(String.format(
                    "Service %s.%s took %dms, accounting for %.1f%% of total trace duration. " +
                    "This is the primary performance bottleneck.",
                    rootCauseRecord.getServiceName(),
                    rootCauseRecord.getMethodName(),
                    rootCauseRecord.getDurationMs() != null ? rootCauseRecord.getDurationMs() : 0,
                    analysis.getImpactPercentage()
                ));
            }
            
            // Find affected downstream services
            List<String> affected = new ArrayList<>();
            int rootIdx = records.indexOf(rootCauseRecord);
            for (int i = rootIdx + 1; i < records.size(); i++) {
                String svc = records.get(i).getServiceName();
                if (!affected.contains(svc)) {
                    affected.add(svc);
                }
            }
            analysis.setAffectedDownstreamServices(affected);
            
            // Collect suspect spans (all slow/error spans)
            List<TraceRecord> suspects = new ArrayList<>();
            suspects.addAll(errorSpans);
            suspects.addAll(slowSpans);
            analysis.setSuspectSpans(suspects.stream().distinct().collect(Collectors.toList()));
            
            // Generate diagnosis steps
            analysis.setDiagnosisSteps(generateDiagnosisSteps(rootCauseRecord, rootCauseType, records));
        } else {
            analysis.setRootCauseFound(false);
            analysis.setRootCauseDetail("No significant bottleneck or error found in this trace.");
        }
        
        return analysis;
    }
    
    /**
     * Generate step-by-step diagnosis suggestions based on root cause
     */
    private List<String> generateDiagnosisSteps(TraceRecord rootCause, String type, List<TraceRecord> allRecords) {
        List<String> steps = new ArrayList<>();
        
        steps.add(String.format("1. Root cause identified: %s.%s (%s)", 
            rootCause.getServiceName(), rootCause.getMethodName(), type));
        
        if ("ERROR".equals(type)) {
            steps.add("2. Check error logs for " + rootCause.getServiceName() + " service");
            steps.add("3. Verify if the error is caused by downstream dependency failure");
            steps.add("4. Check if retry logic is properly configured for this call");
            steps.add("5. Consider adding circuit breaker to prevent cascading failures");
        } else {
            steps.add("2. Check if " + rootCause.getServiceName() + " is experiencing high load");
            steps.add("3. Review database queries and external API calls in " + rootCause.getMethodName());
            steps.add("4. Check if caching can be applied to reduce latency");
            steps.add("5. Consider async processing for non-critical operations");
        }
        
        // Check for error cascade
        long errorCount = allRecords.stream().filter(r -> "ERROR".equals(r.getStatus())).count();
        if (errorCount > 1) {
            steps.add(String.format("6. WARNING: %d errors detected in trace - possible cascade failure", errorCount));
        }
        
        // Check for timeout patterns
        long slowCount = allRecords.stream()
                .filter(r -> r.getDurationMs() != null && r.getDurationMs() > 3000)
                .count();
        if (slowCount > 1) {
            steps.add(String.format("7. WARNING: %d slow spans detected - systemic latency issue", slowCount));
        }
        
        return steps;
    }
    
    /**
     * Clean up old trace records (keep last 10000 records)
     */
    public void cleanupOldTraces() {
        if (traceRecords.size() > 10000) {
            // Remove oldest records keeping the most recent 8000
            List<String> keys = traceRecords.entrySet().stream()
                    .sorted(Comparator.comparingLong(e -> e.getValue().getStartTime()))
                    .limit(2000)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());
            keys.forEach(traceRecords::remove);
        }
        
        // Also clean up trace span order maps
        if (traceSpanOrder.size() > 5000) {
            traceSpanOrder.clear();
        }
    }
    
    /**
     * Internal class for tracking service call statistics
     */
    private static class ServiceCallStats {
        final String sourceService;
        final String targetService;
        int callCount;
        long totalDurationMs;
        int errorCount;
        
        ServiceCallStats(String sourceService, String targetService) {
            this.sourceService = sourceService;
            this.targetService = targetService;
        }
        
        void incrementCalls() { callCount++; }
        
        void incrementErrors() { errorCount++; }
        
        void recordDuration(long durationMs) { totalDurationMs += durationMs; }
    }
    
    /**
     * Internal class for per-service statistics
     */
    private static class ServiceStats {
        final String serviceName;
        int callCount;
        long totalDurationMs;
        long maxDurationMs;
        int errorCount;
        
        ServiceStats(String serviceName) {
            this.serviceName = serviceName;
        }
        
        void incrementCalls() { callCount++; }
        
        void incrementErrors() { errorCount++; }
        
        void recordDuration(long durationMs) {
            totalDurationMs += durationMs;
            if (durationMs > maxDurationMs) {
                maxDurationMs = durationMs;
            }
        }
    }
}
