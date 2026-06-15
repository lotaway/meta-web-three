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
class ContentBasedFilteringServiceTest {

    @Mock
    private UserBehaviorRepository userBehaviorRepository;

    @Mock
    private ProductSimilarityRepository productSimilarityRepository;

    @Mock
    private RecommendationAlgorithmProperties algorithmProperties;

    @InjectMocks
    private ContentBasedFilteringService service;

    @Test
    void contentBasedFiltering_withPurchaseHistory_shouldReturnRecommendations() {
        Long userId = 1L;
        int limit = 5;

        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);

        UserBehavior purchase = new UserBehavior();
        purchase.setUserId(1L);
        purchase.setProductId(100L);
        purchase.setBehaviorType(UserBehavior.BehaviorType.PURCHASE);

        ProductSimilarity sim = new ProductSimilarity();
        sim.setProductId1(100L);
        sim.setProductId2(200L);
        sim.setSimilarityScore(0.75);

        when(userBehaviorRepository.findByUserIdAndBehaviorTypeOrderByTimestampDesc(
            eq(userId), eq(UserBehavior.BehaviorType.PURCHASE)))
            .thenReturn(List.of(purchase));
        when(productSimilarityRepository.findSimilarProductsByAlgorithm(
            eq(100L), eq(ProductSimilarity.SimilarityAlgorithm.CONTENT_BASED)))
            .thenReturn(List.of(sim));

        List<RecommendationResult> results = service.contentBasedFiltering(userId, limit);

        assertNotNull(results);
        assertFalse(results.isEmpty());
        assertEquals(1, results.size());
        RecommendationResult r = results.get(0);
        assertEquals(userId, r.getUserId());
        assertEquals(200L, r.getProductId());
        assertEquals(0.75, r.getScore(), 0.001);
        assertEquals(RecommendationResult.RecommendationAlgorithm.CONTENT_BASED, r.getAlgorithm());
    }

    @Test
    void contentBasedFiltering_withNoPurchaseHistory_shouldReturnEmpty() {
        Long userId = 1L;
        int limit = 5;

        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(userBehaviorRepository.findByUserIdAndBehaviorTypeOrderByTimestampDesc(
            eq(userId), eq(UserBehavior.BehaviorType.PURCHASE)))
            .thenReturn(List.of());

        List<RecommendationResult> results = service.contentBasedFiltering(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }

    @Test
    void contentBasedFiltering_withNoSimilarProducts_shouldReturnEmpty() {
        Long userId = 1L;
        int limit = 5;

        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);

        UserBehavior purchase = new UserBehavior();
        purchase.setUserId(1L);
        purchase.setProductId(100L);
        purchase.setBehaviorType(UserBehavior.BehaviorType.PURCHASE);

        when(userBehaviorRepository.findByUserIdAndBehaviorTypeOrderByTimestampDesc(
            eq(userId), eq(UserBehavior.BehaviorType.PURCHASE)))
            .thenReturn(List.of(purchase));
        when(productSimilarityRepository.findSimilarProductsByAlgorithm(
            eq(100L), eq(ProductSimilarity.SimilarityAlgorithm.CONTENT_BASED)))
            .thenReturn(List.of());

        List<RecommendationResult> results = service.contentBasedFiltering(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }
}
