package com.metawebthree.recommendation.interfaces.controller;

import com.metawebthree.recommendation.application.command.RecommendationCommandService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/recommendation")
public class RecommendationController {

    private final RecommendationCommandService commandService;
    private final RecommendationQueryService queryService;

    public RecommendationController(
            RecommendationCommandService commandService,
            RecommendationQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @PostMapping("/generate")
    public ResponseEntity<Map<String, Object>> generate(
            @RequestBody Map<String, Object> request) {
        Long userId = ((Number) request.get("userId")).longValue();
        String scene = (String) request.get("scene");
        String algorithm = (String) request.get("algorithm");
        int maxItems = (Integer) request.getOrDefault("maxItems", 10);
        
        Recommendation.RecommendationAlgorithm algo = 
            Recommendation.RecommendationAlgorithm.valueOf(algorithm.toUpperCase());
        
        Long recommendationId = commandService.generateRecommendation(
            userId, scene, algo, maxItems);
        
        return ResponseEntity.ok(Map.of("recommendationId", recommendationId));
    }

    @GetMapping("/{id}")
    public ResponseEntity<?> getRecommendation(@PathVariable Long id) {
        return queryService.getRecommendationById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<?> getUserRecommendations(@PathVariable Long userId) {
        return ResponseEntity.ok(queryService.getUserRecommendations(userId));
    }

    @GetMapping("/user/{userId}/scene/{scene}")
    public ResponseEntity<?> getUserRecommendationsByScene(
            @PathVariable Long userId, @PathVariable String scene) {
        return ResponseEntity.ok(queryService.getUserRecommendationsByScene(userId, scene));
    }

    @PostMapping("/behavior")
    public ResponseEntity<Void> recordBehavior(@RequestBody Map<String, Object> request) {
        Long userId = ((Number) request.get("userId")).longValue();
        String skuCode = (String) request.get("skuCode");
        String behaviorType = (String) request.get("behaviorType");
        commandService.recordBehavior(userId, skuCode, behaviorType);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/rule")
    public ResponseEntity<Map<String, Object>> createRule(
            @RequestBody Map<String, Object> request) {
        String ruleName = (String) request.get("ruleName");
        String scene = (String) request.get("scene");
        String type = (String) request.get("type");
        
        RecommendationRule.RuleType ruleType = 
            RecommendationRule.RuleType.valueOf(type.toUpperCase());
        
        Long ruleId = commandService.createRule(ruleName, scene, ruleType);
        
        return ResponseEntity.ok(Map.of("ruleId", ruleId));
    }

    @PostMapping("/rule/{id}/activate")
    public ResponseEntity<Void> activateRule(@PathVariable Long id) {
        commandService.activateRule(id);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/rule/{id}")
    public ResponseEntity<Void> deleteRule(@PathVariable Long id) {
        commandService.deleteRule(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/rule/scene/{scene}")
    public ResponseEntity<?> getRulesByScene(@PathVariable String scene) {
        return ResponseEntity.ok(queryService.getRulesByScene(scene));
    }
}