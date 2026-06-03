package com.metawebthree.common.aitest;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for AI-assisted testing and optimization
 * 
 * Endpoints:
 * - GET  /ai-test/scenarios         - Generate test scenarios
 * - GET  /ai-test/diagnosis/{traceId} - AI root cause diagnosis
 * - GET  /ai-test/suggestions       - Generate optimization suggestions
 * - POST /ai-test/test-flow         - Generate natural language test flow
 */
@RestController
@RequestMapping("/ai-test")
public class AiTestController {
    
    @Autowired
    private AiTestService aiTestService;
    
    /**
     * Generate performance test scenarios
     * @param service Target service name (optional, defaults to all)
     * @param category Test category: PERFORMANCE, STRESS, RELIABILITY, FUNCTIONAL (optional)
     */
    @GetMapping("/scenarios")
    public ResponseEntity<?> generateScenarios(
            @RequestParam(value = "service", required = false) String service,
            @RequestParam(value = "category", required = false) String category) {
        
        List<TestScenario> scenarios = aiTestService.generatePerformanceTestScenarios(service, category);
        return ResponseEntity.ok(Map.of(
            "count", scenarios.size(),
            "scenarios", scenarios
        ));
    }
    
    /**
     * Perform AI root cause diagnosis for a trace
     */
    @GetMapping("/diagnosis/{traceId}")
    public ResponseEntity<?> diagnose(@PathVariable("traceId") String traceId) {
        AiDiagnosisResult result = aiTestService.performAiDiagnosis(traceId);
        return ResponseEntity.ok(result);
    }
    
    /**
     * Generate optimization suggestions based on historical data
     * @param service Target service name (optional)
     */
    @GetMapping("/suggestions")
    public ResponseEntity<?> getSuggestions(
            @RequestParam(value = "service", required = false) String service) {
        
        List<OptimizationSuggestion> suggestions = aiTestService.generateOptimizationSuggestions(service);
        return ResponseEntity.ok(Map.of(
            "count", suggestions.size(),
            "suggestions", suggestions
        ));
    }
    
    /**
     * Generate natural language test flow from a scenario
     * Accepts a scenario in request body and returns readable test procedure
     */
    @PostMapping("/test-flow")
    public ResponseEntity<?> generateTestFlow(@RequestBody TestScenario scenario) {
        String testFlow = aiTestService.generateNaturalLanguageTestFlow(scenario);
        return ResponseEntity.ok(Map.of(
            "scenarioName", scenario.getName(),
            "testFlow", testFlow
        ));
    }
    
    /**
     * Get performance history for a service
     */
    @GetMapping("/history/{service}")
    public ResponseEntity<?> getPerformanceHistory(@PathVariable("service") String service) {
        AiTestService.ServicePerformanceHistory history = aiTestService.getPerformanceHistory(service);
        if (history == null) {
            return ResponseEntity.ok(Map.of(
                "service", service,
                "message", "No performance history available yet"
            ));
        }
        return ResponseEntity.ok(history);
    }
}
