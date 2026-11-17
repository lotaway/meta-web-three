package com.metawebthree.gateway.client;

import com.metawebthree.common.generated.rpc.ActivateRuleRequest;
import com.metawebthree.common.generated.rpc.CreateRuleRequest;
import com.metawebthree.common.generated.rpc.CreateRuleResponse;
import com.metawebthree.common.generated.rpc.DeleteRuleRequest;
import com.metawebthree.common.generated.rpc.GenerateRecommendationRequest;
import com.metawebthree.common.generated.rpc.GenerateRecommendationResponse;
import com.metawebthree.common.generated.rpc.GenerateRecommendationsRequest;
import com.metawebthree.common.generated.rpc.GetBehaviorHistoryRequest;
import com.metawebthree.common.generated.rpc.GetBehaviorHistoryResponse;
import com.metawebthree.common.generated.rpc.GetMetricsRequest;
import com.metawebthree.common.generated.rpc.GetMetricsResponse;
import com.metawebthree.common.generated.rpc.GetRecommendationByIdRequest;
import com.metawebthree.common.generated.rpc.GetRecommendationByIdResponse;
import com.metawebthree.common.generated.rpc.GetRecommendationsByAlgorithmRequest;
import com.metawebthree.common.generated.rpc.GetRecommendationsByAlgorithmResponse;
import com.metawebthree.common.generated.rpc.GetRulesBySceneRequest;
import com.metawebthree.common.generated.rpc.GetRulesBySceneResponse;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsBySceneRequest;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsBySceneResponse;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsRequest;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsResponse;
import com.metawebthree.common.generated.rpc.MarkRecommendationRequest;
import com.metawebthree.common.generated.rpc.RecommendationResultView;
import com.metawebthree.common.generated.rpc.RecommendationRuleView;
import com.metawebthree.common.generated.rpc.RecommendationService;
import com.metawebthree.common.generated.rpc.RecommendationView;
import com.metawebthree.common.generated.rpc.RecordBehaviorRequest;
import com.metawebthree.common.generated.rpc.UserBehaviorView;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gateway-side client for mall-catalog recommendations. Calls the service over
 * Dubbo instead of raw HTTP; the public method signatures still return maps so
 * the GraphQL data fetchers keep working unchanged.
 */
@Slf4j
@Component
public class RecommendationClient {

    @DubboReference(check = false, lazy = true)
    private RecommendationService recommendationService;

    public Map<String, Object> generateRecommendation(Long userId, String scene, String algorithm, int maxItems) {
        GenerateRecommendationResponse response = recommendationService.generateRecommendation(
                GenerateRecommendationRequest.newBuilder()
                        .setUserId(userId).setScene(scene).setAlgorithm(algorithm).setMaxItems(maxItems)
                        .build());
        return response.hasRecommendation() ? recommendationToMap(response.getRecommendation()) : null;
    }

    public Map<String, Object> getRecommendationById(Long id) {
        GetRecommendationByIdResponse response = recommendationService.getRecommendationById(
                GetRecommendationByIdRequest.newBuilder().setId(id).build());
        return response.hasRecommendation() ? recommendationToMap(response.getRecommendation()) : null;
    }

    public List<Map<String, Object>> getUserRecommendations(Long userId) {
        GetUserRecommendationsResponse response = recommendationService.getUserRecommendations(
                GetUserRecommendationsRequest.newBuilder().setUserId(userId).build());
        List<Map<String, Object>> results = new ArrayList<>();
        for (RecommendationView view : response.getRecommendationsList()) {
            results.add(recommendationToMap(view));
        }
        return results;
    }

    public List<Map<String, Object>> getUserRecommendationsByScene(Long userId, String scene) {
        GetUserRecommendationsBySceneResponse response = recommendationService.getUserRecommendationsByScene(
                GetUserRecommendationsBySceneRequest.newBuilder().setUserId(userId).setScene(scene).build());
        List<Map<String, Object>> results = new ArrayList<>();
        for (RecommendationView view : response.getRecommendationsList()) {
            results.add(recommendationToMap(view));
        }
        return results;
    }

    public void recordBehavior(Long userId, String skuCode, String behaviorType) {
        recommendationService.recordBehavior(RecordBehaviorRequest.newBuilder()
                .setUserId(userId).setSkuCode(skuCode).setBehaviorType(behaviorType).build());
    }

    public Map<String, Object> createRule(String ruleName, String scene, String type) {
        CreateRuleResponse response = recommendationService.createRule(CreateRuleRequest.newBuilder()
                .setRuleName(ruleName).setScene(scene).setType(type).build());
        return response.hasRule() ? ruleToMap(response.getRule()) : null;
    }

    public void activateRule(Long id) {
        recommendationService.activateRule(ActivateRuleRequest.newBuilder().setId(id).build());
    }

    public void deleteRule(Long id) {
        recommendationService.deleteRule(DeleteRuleRequest.newBuilder().setId(id).build());
    }

    public List<Map<String, Object>> getRulesByScene(String scene) {
        GetRulesBySceneResponse response = recommendationService.getRulesByScene(
                GetRulesBySceneRequest.newBuilder().setScene(scene).build());
        List<Map<String, Object>> results = new ArrayList<>();
        for (RecommendationRuleView view : response.getRulesList()) {
            results.add(ruleToMap(view));
        }
        return results;
    }

    public List<Map<String, Object>> getRecommendationsByAlgorithm(Long userId, String algorithm, int limit) {
        GetRecommendationsByAlgorithmResponse response = recommendationService.getRecommendationsByAlgorithm(
                GetRecommendationsByAlgorithmRequest.newBuilder()
                        .setUserId(userId).setAlgorithm(algorithm).setLimit(limit).build());
        return resultViewsToMaps(response.getResultsList());
    }

    public List<Map<String, Object>> generateRecommendations(Long userId, int limit) {
        GetRecommendationsByAlgorithmResponse response = recommendationService.generateRecommendations(
                GenerateRecommendationsRequest.newBuilder().setUserId(userId).setLimit(limit).build());
        return resultViewsToMaps(response.getResultsList());
    }

    public List<Map<String, Object>> getBehaviorHistory(Long userId, int limit) {
        GetBehaviorHistoryResponse response = recommendationService.getBehaviorHistory(
                GetBehaviorHistoryRequest.newBuilder().setUserId(userId).setLimit(limit).build());
        List<Map<String, Object>> results = new ArrayList<>();
        for (UserBehaviorView view : response.getBehaviorsList()) {
            results.add(behaviorToMap(view));
        }
        return results;
    }

    public void markAsClicked(Long recommendationId) {
        recommendationService.markAsClicked(
                MarkRecommendationRequest.newBuilder().setRecommendationId(recommendationId).build());
    }

    public void markAsPurchased(Long recommendationId) {
        recommendationService.markAsPurchased(
                MarkRecommendationRequest.newBuilder().setRecommendationId(recommendationId).build());
    }

    public Map<String, Object> getMetrics(Long userId) {
        GetMetricsResponse response = recommendationService.getMetrics(
                GetMetricsRequest.newBuilder().setUserId(userId).build());
        if (!response.hasMetrics()) {
            return Map.of();
        }
        Map<String, Object> map = new HashMap<>();
        map.put("totalRecommendations", response.getMetrics().getTotalRecommendations());
        map.put("clickedCount", response.getMetrics().getClickedCount());
        map.put("purchasedCount", response.getMetrics().getPurchasedCount());
        map.put("clickThroughRate", response.getMetrics().getClickThroughRate());
        map.put("conversionRate", response.getMetrics().getConversionRate());
        return map;
    }

    /**
     * Paginated read kept for source compatibility. The GraphQL layer uses the
     * plain list reads; this maps the full list into a simple paginated shape.
     */
    public Map<String, Object> getRecommendationsPaginated(Long userId, int page, int size) {
        List<Map<String, Object>> all = getUserRecommendations(userId);
        int from = Math.max(0, page * size);
        int to = Math.min(all.size(), from + size);
        Map<String, Object> result = new HashMap<>();
        result.put("content", from <= to ? all.subList(from, to) : List.of());
        result.put("totalElements", all.size());
        result.put("page", page);
        result.put("size", size);
        return result;
    }

    private List<Map<String, Object>> resultViewsToMaps(List<RecommendationResultView> views) {
        List<Map<String, Object>> results = new ArrayList<>();
        for (RecommendationResultView view : views) {
            results.add(resultToMap(view));
        }
        return results;
    }

    private Map<String, Object> recommendationToMap(RecommendationView v) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", v.getId());
        map.put("userId", v.getUserId());
        map.put("scene", v.getScene());
        map.put("algorithm", v.getAlgorithm());
        map.put("score", v.getScore());
        map.put("status", v.getStatus());
        map.put("clickCount", v.getClickCount());
        map.put("conversionCount", v.getConversionCount());
        map.put("impressionCount", v.getImpressionCount());
        map.put("createdAt", v.getCreatedAt());
        map.put("expiresAt", v.getExpiresAt());
        List<Map<String, Object>> items = new ArrayList<>();
        for (var item : v.getItemsList()) {
            Map<String, Object> im = new HashMap<>();
            im.put("skuCode", item.getSkuCode());
            im.put("skuName", item.getSkuName());
            im.put("score", item.getScore());
            im.put("rank", item.getRank());
            im.put("reason", item.getReason());
            items.add(im);
        }
        map.put("items", items);
        return map;
    }

    private Map<String, Object> resultToMap(RecommendationResultView v) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", v.getId());
        map.put("userId", v.getUserId());
        map.put("productId", v.getProductId());
        map.put("score", v.getScore());
        map.put("algorithm", v.getAlgorithm());
        map.put("reason", v.getReason());
        map.put("position", v.getPosition());
        map.put("isClicked", v.getIsClicked());
        map.put("isPurchased", v.getIsPurchased());
        map.put("createdAt", v.getCreatedAt());
        map.put("expiresAt", v.getExpiresAt());
        return map;
    }

    private Map<String, Object> ruleToMap(RecommendationRuleView v) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", v.getId());
        map.put("ruleName", v.getRuleName());
        map.put("scene", v.getScene());
        map.put("type", v.getType());
        map.put("status", v.getStatus());
        map.put("priority", v.getPriority());
        map.put("maxItems", v.getMaxItems());
        map.put("conditions", v.getConditions());
        map.put("exclusions", v.getExclusions());
        map.put("boostFactor", v.getBoostFactor());
        map.put("createdAt", v.getCreatedAt());
        map.put("updatedAt", v.getUpdatedAt());
        return map;
    }

    private Map<String, Object> behaviorToMap(UserBehaviorView v) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", v.getId());
        map.put("userId", v.getUserId());
        map.put("productId", v.getProductId());
        map.put("behaviorType", v.getBehaviorType());
        map.put("behaviorValue", v.getBehaviorValue());
        map.put("timestamp", v.getTimestamp());
        map.put("sessionId", v.getSessionId());
        map.put("source", v.getSource());
        return map;
    }
}
