package com.metawebthree.gateway.client;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.client.loadbalancer.LoadBalanced;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpMethod;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Slf4j
@Component
public class RecommendationClient {

    private final String serviceUrl;
    private final RestTemplate restTemplate;

    public RecommendationClient(
            @Value("${recommendation.service.url:http://recommendation-service/api/recommendation}") String serviceUrl,
            @LoadBalanced RestTemplate restTemplate) {
        this.serviceUrl = serviceUrl;
        this.restTemplate = restTemplate;
    }

    public Map<String, Object> generateRecommendation(Long userId, String scene, String algorithm, int maxItems) {
        Map<String, Object> request = new HashMap<>();
        request.put("userId", userId);
        request.put("scene", scene);
        request.put("algorithm", algorithm);
        request.put("maxItems", maxItems);
        return restTemplate.postForObject(serviceUrl + "/generate", request, Map.class);
    }

    public Map<String, Object> getRecommendationById(Long id) {
        return restTemplate.getForObject(serviceUrl + "/" + id, Map.class);
    }

    public List<Map<String, Object>> getUserRecommendations(Long userId) {
        String url = serviceUrl + "/user/" + userId;
        return restTemplate.exchange(url, HttpMethod.GET, null,
            new ParameterizedTypeReference<List<Map<String, Object>>>() {}).getBody();
    }

    public List<Map<String, Object>> getUserRecommendationsByScene(Long userId, String scene) {
        String url = serviceUrl + "/user/" + userId + "/scene/" + scene;
        return restTemplate.exchange(url, HttpMethod.GET, null,
            new ParameterizedTypeReference<List<Map<String, Object>>>() {}).getBody();
    }

    public void recordBehavior(Long userId, String skuCode, String behaviorType) {
        Map<String, Object> request = new HashMap<>();
        request.put("userId", userId);
        request.put("skuCode", skuCode);
        request.put("behaviorType", behaviorType);
        restTemplate.postForObject(serviceUrl + "/behavior", request, Void.class);
    }

    public Map<String, Object> createRule(String ruleName, String scene, String type) {
        Map<String, Object> request = new HashMap<>();
        request.put("ruleName", ruleName);
        request.put("scene", scene);
        request.put("type", type);
        return restTemplate.postForObject(serviceUrl + "/rule", request, Map.class);
    }

    public void activateRule(Long id) {
        restTemplate.postForObject(serviceUrl + "/rule/" + id + "/activate", null, Void.class);
    }

    public void deleteRule(Long id) {
        restTemplate.delete(serviceUrl + "/rule/" + id);
    }

    public List<Map<String, Object>> getRulesByScene(String scene) {
        String url = serviceUrl + "/rule/scene/" + scene;
        return restTemplate.exchange(url, HttpMethod.GET, null,
            new ParameterizedTypeReference<List<Map<String, Object>>>() {}).getBody();
    }

    public List<Map<String, Object>> getRecommendationsByAlgorithm(Long userId, String algorithm, int limit) {
        String url = serviceUrl + "/user/" + userId + "/algorithm/" + algorithm + "?limit=" + limit;
        return restTemplate.exchange(url, HttpMethod.GET, null,
            new ParameterizedTypeReference<List<Map<String, Object>>>() {}).getBody();
    }

    public List<Map<String, Object>> generateRecommendations(Long userId, int limit) {
        String url = serviceUrl + "/user/" + userId + "/generate?limit=" + limit;
        return restTemplate.exchange(url, HttpMethod.POST, null,
            new ParameterizedTypeReference<List<Map<String, Object>>>() {}).getBody();
    }

    public List<Map<String, Object>> getBehaviorHistory(Long userId, int limit) {
        String url = serviceUrl + "/behavior/user/" + userId + "?limit=" + limit;
        return restTemplate.exchange(url, HttpMethod.GET, null,
            new ParameterizedTypeReference<List<Map<String, Object>>>() {}).getBody();
    }

    public void markAsClicked(Long recommendationId) {
        restTemplate.postForObject(serviceUrl + "/" + recommendationId + "/click", null, Void.class);
    }

    public void markAsPurchased(Long recommendationId) {
        restTemplate.postForObject(serviceUrl + "/" + recommendationId + "/purchase", null, Void.class);
    }

    public Map<String, Object> getMetrics(Long userId) {
        return restTemplate.getForObject(serviceUrl + "/metrics/user/" + userId, Map.class);
    }

    public Map<String, Object> getRecommendationsPaginated(Long userId, int page, int size) {
        String url = serviceUrl + "/user/" + userId + "/paginated?page=" + page + "&size=" + size;
        return restTemplate.getForObject(url, Map.class);
    }
}
