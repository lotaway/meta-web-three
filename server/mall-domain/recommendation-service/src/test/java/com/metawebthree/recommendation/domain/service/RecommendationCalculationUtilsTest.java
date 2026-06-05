package com.metawebthree.recommendation.domain.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class RecommendationCalculationUtilsTest {

    @Test
    void createRecommendationResult_shouldSetCorrectFields() {
        RecommendationResult result = RecommendationCalculationUtils.createRecommendationResult(
            1L, 100L, 85.5, RecommendationResult.RecommendationAlgorithm.HYBRID, 7);

        assertEquals(1L, result.getUserId());
        assertEquals(100L, result.getProductId());
        assertEquals(85.5, result.getScore(), 0.001);
        assertEquals(RecommendationResult.RecommendationAlgorithm.HYBRID, result.getAlgorithm());
        assertFalse(result.getIsClicked());
        assertFalse(result.getIsPurchased());
        assertNotNull(result.getCreatedAt());
        assertNotNull(result.getExpiresAt());
    }

    @Test
    void combineRecommendations_shouldMergeCorrectly() {
        Map<Long, Double> combined = new HashMap<>();
        RecommendationResult r1 = new RecommendationResult();
        r1.setProductId(1L);
        r1.setScore(80.0);
        RecommendationResult r2 = new RecommendationResult();
        r2.setProductId(2L);
        r2.setScore(70.0);

        RecommendationCalculationUtils.combineRecommendations(combined, List.of(r1, r2), 0.5);

        assertEquals(40.0, combined.get(1L), 0.001);
        assertEquals(35.0, combined.get(2L), 0.001);
    }

    @Test
    void combineRecommendations_withDuplicateProducts_shouldSumScores() {
        Map<Long, Double> combined = new HashMap<>();
        RecommendationResult r1 = new RecommendationResult();
        r1.setProductId(1L);
        r1.setScore(80.0);
        RecommendationResult r2 = new RecommendationResult();
        r2.setProductId(1L);
        r2.setScore(20.0);

        RecommendationCalculationUtils.combineRecommendations(combined, List.of(r1, r2), 1.0);

        assertEquals(100.0, combined.get(1L), 0.001);
    }

    @Test
    void calculateJaccardSimilarity_withSharedUsers_shouldReturnPositive() {
        UserBehaviorRepository repo = mock(UserBehaviorRepository.class);
        when(repo.findUserIdsByProductId(1L)).thenReturn(List.of(1L, 2L, 3L));
        when(repo.findUserIdsByProductId(2L)).thenReturn(List.of(2L, 3L, 4L));

        double similarity = RecommendationCalculationUtils.calculateJaccardSimilarity(repo, 1L, 2L);

        assertEquals(2.0 / 4.0, similarity, 0.001);
    }

    @Test
    void calculateJaccardSimilarity_withNoSharedUsers_shouldReturnZero() {
        UserBehaviorRepository repo = mock(UserBehaviorRepository.class);
        when(repo.findUserIdsByProductId(1L)).thenReturn(List.of(1L, 2L));
        when(repo.findUserIdsByProductId(2L)).thenReturn(List.of(3L, 4L));

        double similarity = RecommendationCalculationUtils.calculateJaccardSimilarity(repo, 1L, 2L);

        assertEquals(0.0, similarity, 0.001);
    }

    @Test
    void calculateCosineSimilarity_withIdenticalVectors_shouldReturnOne() {
        Map<Long, Double> v1 = new HashMap<>();
        v1.put(1L, 5.0);
        v1.put(2L, 3.0);
        Map<Long, Double> v2 = new HashMap<>();
        v2.put(1L, 5.0);
        v2.put(2L, 3.0);

        double similarity = RecommendationCalculationUtils.calculateCosineSimilarity(v1, v2);

        assertEquals(1.0, similarity, 0.001);
    }

    @Test
    void calculateCosineSimilarity_withOrthogonalVectors_shouldReturnZero() {
        Map<Long, Double> v1 = new HashMap<>();
        v1.put(1L, 5.0);
        Map<Long, Double> v2 = new HashMap<>();
        v2.put(2L, 3.0);

        double similarity = RecommendationCalculationUtils.calculateCosineSimilarity(v1, v2);

        assertEquals(0.0, similarity, 0.001);
    }

    @Test
    void getBehaviorWeight_shouldReturnCorrectValue() {
        var behavior = new RecommendationAlgorithmProperties.Behavior();
        behavior.setPurchaseWeight(5.0);
        behavior.setCartWeight(4.0);
        behavior.setCollectWeight(3.0);
        behavior.setClickWeight(2.0);
        behavior.setViewWeight(1.0);

        assertEquals(5.0, RecommendationCalculationUtils.getBehaviorWeight(behavior, UserBehavior.BehaviorType.PURCHASE), 0.001);
        assertEquals(4.0, RecommendationCalculationUtils.getBehaviorWeight(behavior, UserBehavior.BehaviorType.CART), 0.001);
        assertEquals(3.0, RecommendationCalculationUtils.getBehaviorWeight(behavior, UserBehavior.BehaviorType.COLLECT), 0.001);
        assertEquals(2.0, RecommendationCalculationUtils.getBehaviorWeight(behavior, UserBehavior.BehaviorType.CLICK), 0.001);
        assertEquals(1.0, RecommendationCalculationUtils.getBehaviorWeight(behavior, UserBehavior.BehaviorType.VIEW), 0.001);
    }

    @Test
    void createBehaviorVector_shouldAggregateWeights() {
        var behavior = new RecommendationAlgorithmProperties.Behavior();
        behavior.setPurchaseWeight(5.0);
        behavior.setViewWeight(1.0);

        UserBehavior b1 = new UserBehavior();
        b1.setProductId(100L);
        b1.setBehaviorType(UserBehavior.BehaviorType.PURCHASE);
        UserBehavior b2 = new UserBehavior();
        b2.setProductId(100L);
        b2.setBehaviorType(UserBehavior.BehaviorType.VIEW);

        Map<Long, Double> vector = RecommendationCalculationUtils.createBehaviorVector(List.of(b1, b2), behavior);

        assertEquals(6.0, vector.get(100L), 0.001);
    }
}
