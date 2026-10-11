package com.metawebthree.recommendation.interfaces.controller;

import com.metawebthree.recommendation.application.command.RecommendationCommandService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService.RecommendationMetrics;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
    public ResponseEntity<Recommendation> generate(
            @RequestBody Map<String, Object> request) {
        Long userId = getLongRequired(request, "userId");
        String scene = getStringRequired(request, "scene");
        String algorithm = getStringRequired(request, "algorithm");
        int maxItems = getIntOrDefault(request, "maxItems", 10);

        Recommendation.RecommendationAlgorithm algo =
            Recommendation.RecommendationAlgorithm.valueOf(algorithm.toUpperCase());

        Recommendation recommendation = commandService.generateRecommendation(
            userId, scene, algo, maxItems);

        return ResponseEntity.ok(recommendation);
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
        Long userId = getLongRequired(request, "userId");
        String skuCode = getStringRequired(request, "skuCode");
        String behaviorType = getStringRequired(request, "behaviorType");
        commandService.recordBehavior(userId, skuCode, behaviorType);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/rule")
    public ResponseEntity<RecommendationRule> createRule(
            @RequestBody Map<String, Object> request) {
        String ruleName = getStringRequired(request, "ruleName");
        String scene = getStringRequired(request, "scene");
        String type = getStringRequired(request, "type");

        RecommendationRule.RuleType ruleType =
            RecommendationRule.RuleType.valueOf(type.toUpperCase());

        RecommendationRule rule = commandService.createRule(ruleName, scene, ruleType);

        return ResponseEntity.ok(rule);
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

    @GetMapping("/user/{userId}/algorithm/{algorithm}")
    public ResponseEntity<List<RecommendationResult>> getRecommendationsByAlgorithm(
            @PathVariable Long userId,
            @PathVariable String algorithm,
            @RequestParam(defaultValue = "10") int limit) {
        RecommendationResult.RecommendationAlgorithm algo =
            RecommendationResult.RecommendationAlgorithm.valueOf(algorithm.toUpperCase());
        List<RecommendationResult> recommendations =
            queryService.getRecommendationsByAlgorithm(userId, algo, limit);
        return ResponseEntity.ok(recommendations);
    }

    @PostMapping("/user/{userId}/generate")
    public ResponseEntity<List<RecommendationResult>> generateRecommendations(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "10") int limit) {
        List<RecommendationResult> recommendations =
            commandService.generateRecommendationsByAlgorithm(userId, null, limit);
        return ResponseEntity.ok(recommendations);
    }

    @GetMapping("/behavior/user/{userId}")
    public ResponseEntity<List<UserBehavior>> getBehaviorHistory(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "50") int limit) {
        List<UserBehavior> behaviors = queryService.getUserBehaviorHistory(userId, limit);
        return ResponseEntity.ok(behaviors);
    }

    @PostMapping("/{recommendationId}/click")
    public ResponseEntity<Void> markRecommendationClicked(@PathVariable Long recommendationId) {
        commandService.markRecommendationClicked(recommendationId);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{recommendationId}/purchase")
    public ResponseEntity<Void> markRecommendationPurchased(@PathVariable Long recommendationId) {
        commandService.markRecommendationPurchased(recommendationId);
        return ResponseEntity.ok().build();
    }

    @SuppressWarnings("unchecked")
    @PostMapping("/purchase-by-product")
    public ResponseEntity<Void> markPurchasedByProduct(@RequestBody Map<String, Object> request) {
        Long userId = getLongRequired(request, "userId");
        Long productId = getLongRequired(request, "productId");
        commandService.markPurchasedByProduct(userId, productId);
        return ResponseEntity.ok().build();
    }

    @SuppressWarnings("unchecked")
    @PostMapping("/purchase-by-products")
    public ResponseEntity<Void> markPurchasedByProducts(@RequestBody Map<String, Object> request) {
        Long userId = getLongRequired(request, "userId");
        List<Object> productIdObjs = (List<Object>) request.get("productIds");
        List<Long> productIds = productIdObjs.stream()
                .map(obj -> ((Number) obj).longValue())
                .collect(Collectors.toList());
        commandService.markPurchasedByProducts(userId, productIds);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/metrics/user/{userId}")
    public ResponseEntity<RecommendationMetrics> getMetrics(@PathVariable Long userId) {
        RecommendationMetrics metrics = queryService.getRecommendationMetrics(userId);
        return ResponseEntity.ok(metrics);
    }

    @GetMapping("/user/{userId}/paginated")
    public ResponseEntity<Page<RecommendationResult>> getRecommendationsPaginated(
            @PathVariable Long userId,
            @PageableDefault(size = 20) Pageable pageable) {
        Page<RecommendationResult> page = queryService.getRecommendationsPaginated(userId, pageable);
        return ResponseEntity.ok(page);
    }

    @PostMapping("/admin/update-similarity")
    public ResponseEntity<String> updateSimilarityMatrix() {
        commandService.updateProductSimilarities();
        return ResponseEntity.ok("Similarity matrix update triggered");
    }

    @PostMapping("/admin/cleanup")
    public ResponseEntity<String> cleanupOldData(
            @RequestParam(defaultValue = "90") int daysToKeep) {
        commandService.cleanupOldData(daysToKeep);
        return ResponseEntity.ok("Cleanup completed");
    }

    private Long getLongRequired(Map<String, Object> request, String key) {
        Object value = request.get(key);
        if (value == null) {
            throw new IllegalArgumentException("Missing required field: " + key);
        }
        return ((Number) value).longValue();
    }

    private String getStringRequired(Map<String, Object> request, String key) {
        Object value = request.get(key);
        if (value == null) {
            throw new IllegalArgumentException("Missing required field: " + key);
        }
        return (String) value;
    }

    private int getIntOrDefault(Map<String, Object> request, String key, int defaultValue) {
        Object value = request.get(key);
        if (value == null) {
            return defaultValue;
        }
        return ((Integer) value);
    }
}
