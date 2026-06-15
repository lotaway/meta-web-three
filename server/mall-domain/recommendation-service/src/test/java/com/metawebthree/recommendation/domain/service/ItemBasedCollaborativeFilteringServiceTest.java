package com.metawebthree.recommendation.domain.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class ItemBasedCollaborativeFilteringServiceTest {

    @Mock
    private UserBehaviorRepository userBehaviorRepository;

    @Mock
    private ProductSimilarityRepository productSimilarityRepository;

    @Mock
    private RecommendationAlgorithmProperties algorithmProperties;

    @InjectMocks
    private ItemBasedCollaborativeFilteringService service;

    @Test
    void itemBasedCollaborativeFiltering_withBehaviorsAndSimilarities_shouldReturnRecommendations() {
        Long userId = 1L;
        int limit = 5;

        var behavior = new RecommendationAlgorithmProperties.Behavior();
        behavior.setPurchaseWeight(5.0);
        behavior.setViewWeight(1.0);

        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(algorithmProperties.getBehavior()).thenReturn(behavior);

        UserBehavior userBehavior = new UserBehavior();
        userBehavior.setUserId(1L);
        userBehavior.setProductId(100L);
        userBehavior.setBehaviorType(UserBehavior.BehaviorType.PURCHASE);
        userBehavior.setBehaviorValue(5.0);

        ProductSimilarity sim = new ProductSimilarity();
        sim.setProductId1(100L);
        sim.setProductId2(200L);
        sim.setSimilarityScore(0.8);

        when(userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId)).thenReturn(List.of(userBehavior));
        when(productSimilarityRepository.findSimilarProducts(100L)).thenReturn(List.of(sim));

        List<RecommendationResult> results = service.itemBasedCollaborativeFiltering(userId, limit);

        assertNotNull(results);
        assertFalse(results.isEmpty());
        assertEquals(1, results.size());
        RecommendationResult r = results.get(0);
        assertEquals(userId, r.getUserId());
        assertEquals(200L, r.getProductId());
        assertEquals(4.0, r.getScore(), 0.001);
        assertEquals(RecommendationResult.RecommendationAlgorithm.ITEM_BASED_CF, r.getAlgorithm());
    }

    @Test
    void itemBasedCollaborativeFiltering_withNoBehaviors_shouldReturnEmpty() {
        Long userId = 1L;
        int limit = 5;

        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId)).thenReturn(List.of());

        List<RecommendationResult> results = service.itemBasedCollaborativeFiltering(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }

    @Test
    void itemBasedCollaborativeFiltering_withNoSimilarities_shouldReturnEmpty() {
        Long userId = 1L;
        int limit = 5;

        var behavior = new RecommendationAlgorithmProperties.Behavior();
        behavior.setPurchaseWeight(5.0);

        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(algorithmProperties.getBehavior()).thenReturn(behavior);

        UserBehavior userBehavior = new UserBehavior();
        userBehavior.setUserId(1L);
        userBehavior.setProductId(100L);
        userBehavior.setBehaviorType(UserBehavior.BehaviorType.PURCHASE);

        when(userBehaviorRepository.findByUserIdOrderByTimestampDesc(userId)).thenReturn(List.of(userBehavior));
        when(productSimilarityRepository.findSimilarProducts(100L)).thenReturn(List.of());

        List<RecommendationResult> results = service.itemBasedCollaborativeFiltering(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }
}
