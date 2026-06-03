package com.metawebthree.common.trace;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for distributed trace querying, bottleneck analysis,
 * service topology visualization, and root cause diagnosis
 */
@RestController
@RequestMapping("/trace")
public class TraceController {
    
    @Autowired
    private DistributedTraceService traceService;
    
    /**
     * Get all trace records for a given trace ID (call chain visualization)
     */
    @GetMapping("/{traceId}")
    public ResponseEntity<?> getTraceRecords(@PathVariable("traceId") String traceId) {
        List<TraceRecord> records = traceService.getTraceRecords(traceId);
        return ResponseEntity.ok(records);
    }
    
    /**
     * Analyze trace for performance bottlenecks
     */
    @GetMapping("/{traceId}/analysis")
    public ResponseEntity<?> analyzeTrace(@PathVariable("traceId") String traceId) {
        BottleneckAnalysis analysis = traceService.analyzeBottlenecks(traceId);
        return ResponseEntity.ok(analysis);
    }
    
    /**
     * Perform root cause analysis on a trace
     * Identifies the most likely root cause and provides diagnosis steps
     */
    @GetMapping("/{traceId}/root-cause")
    public ResponseEntity<?> rootCauseAnalysis(@PathVariable("traceId") String traceId) {
        RootCauseAnalysis analysis = traceService.performRootCauseAnalysis(traceId);
        return ResponseEntity.ok(analysis);
    }
    
    /**
     * Get current trace ID (for debugging)
     */
    @GetMapping("/current")
    public ResponseEntity<?> getCurrentTraceId() {
        String traceId = traceService.getTraceId();
        
        Map<String, String> result = Map.of(
            "traceId", traceId,
            "timestamp", String.valueOf(System.currentTimeMillis())
        );
        
        return ResponseEntity.ok(result);
    }
    
    /**
     * Get recent trace IDs
     */
    @GetMapping("/recent")
    public ResponseEntity<?> getRecentTraces(
            @RequestParam(value = "limit", defaultValue = "20") int limit) {
        List<String> traceIds = traceService.getRecentTraceIds(limit);
        return ResponseEntity.ok(traceIds);
    }
    
    /**
     * Get slowest spans across all traces (bottleneck detection)
     * Returns spans that exceed the minimum duration threshold
     */
    @GetMapping("/bottlenecks")
    public ResponseEntity<?> getBottlenecks(
            @RequestParam(value = "minDurationMs", defaultValue = "3000") long minDurationMs,
            @RequestParam(value = "limit", defaultValue = "50") int limit) {
        
        List<TraceRecord> slowSpans = traceService.getSlowSpans(minDurationMs);
        
        // Limit the results
        if (slowSpans.size() > limit) {
            slowSpans = slowSpans.subList(0, limit);
        }
        
        return ResponseEntity.ok(Map.of(
            "minDurationMs", minDurationMs,
            "count", slowSpans.size(),
            "bottlenecks", slowSpans
        ));
    }
    
    /**
     * Get service dependency topology
     * Auto-discovered from actual trace data
     */
    @GetMapping("/topology")
    public ResponseEntity<?> getServiceTopology() {
        ServiceTopology topology = traceService.buildServiceTopology();
        return ResponseEntity.ok(topology);
    }
}
