package com.metawebthree.recommendation.domain.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class HybridRecommendationServiceTest {

    @Mock
    private UserBasedCollaborativeFilteringService userBasedCFService;

    @Mock
    private ItemBasedCollaborativeFilteringService itemBasedCFService;

    @Mock
    private ContentBasedFilteringService contentBasedService;

    @Mock
    private PopularityBasedRecommendationService popularityService;

    @Mock
    private RecommendationAlgorithmProperties algorithmProperties;

    @InjectMocks
    private HybridRecommendationService service;

    @Test
    void hybridRecommendation_withAllAlgorithms_shouldCombineAndReturn() {
        Long userId = 1L;
        int limit = 5;

        var scoring = new RecommendationAlgorithmProperties.Scoring();
        scoring.setCollaborativeWeight(0.4);
        scoring.setContentWeight(0.3);
        scoring.setAiModelWeight(0.5);
        scoring.setPopularityWeight(0.6);
        scoring.setBaseScore(80.0);
        scoring.setScoreDecay(5.0);

        when(algorithmProperties.getScoring()).thenReturn(scoring);
        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);

        RecommendationResult userCfRec = new RecommendationResult();
        userCfRec.setUserId(userId); userCfRec.setProductId(100L); userCfRec.setScore(80.0);
        userCfRec.setAlgorithm(RecommendationResult.RecommendationAlgorithm.USER_BASED_CF);
        when(userBasedCFService.userBasedCollaborativeFiltering(userId, limit * 2))
            .thenReturn(List.of(userCfRec));

        RecommendationResult itemCfRec = new RecommendationResult();
        itemCfRec.setUserId(userId); itemCfRec.setProductId(200L); itemCfRec.setScore(70.0);
        itemCfRec.setAlgorithm(RecommendationResult.RecommendationAlgorithm.ITEM_BASED_CF);
        when(itemBasedCFService.itemBasedCollaborativeFiltering(userId, limit * 2))
            .thenReturn(List.of(itemCfRec));

        RecommendationResult contentRec = new RecommendationResult();
        contentRec.setUserId(userId); contentRec.setProductId(300L); contentRec.setScore(60.0);
        contentRec.setAlgorithm(RecommendationResult.RecommendationAlgorithm.CONTENT_BASED);
        when(contentBasedService.contentBasedFiltering(userId, limit * 2))
            .thenReturn(List.of(contentRec));

        RecommendationResult popularityRec = new RecommendationResult();
        popularityRec.setUserId(userId); popularityRec.setProductId(100L); popularityRec.setScore(50.0);
        popularityRec.setAlgorithm(RecommendationResult.RecommendationAlgorithm.POPULARITY);
        when(popularityService.popularityBasedRecommendation(userId, limit * 2))
            .thenReturn(List.of(popularityRec));

        List<RecommendationResult> results = service.hybridRecommendation(userId, limit);

        assertNotNull(results);
        assertFalse(results.isEmpty());
        assertTrue(results.size() <= limit);
        for (RecommendationResult r : results) {
            assertEquals(userId, r.getUserId());
            assertEquals(RecommendationResult.RecommendationAlgorithm.HYBRID, r.getAlgorithm());
        }
    }

    @Test
    void hybridRecommendation_withNoSubResults_shouldReturnEmpty() {
        Long userId = 1L;
        int limit = 5;

        var scoring = new RecommendationAlgorithmProperties.Scoring();
        scoring.setCollaborativeWeight(0.4);
        scoring.setContentWeight(0.3);
        scoring.setAiModelWeight(0.5);
        scoring.setPopularityWeight(0.6);

        when(algorithmProperties.getScoring()).thenReturn(scoring);
        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(userBasedCFService.userBasedCollaborativeFiltering(userId, limit * 2)).thenReturn(List.of());
        when(itemBasedCFService.itemBasedCollaborativeFiltering(userId, limit * 2)).thenReturn(List.of());
        when(contentBasedService.contentBasedFiltering(userId, limit * 2)).thenReturn(List.of());
        when(popularityService.popularityBasedRecommendation(userId, limit * 2)).thenReturn(List.of());

        List<RecommendationResult> results = service.hybridRecommendation(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }

    @Test
    void hybridRecommendation_duplicateProductsAcrossAlgorithms_shouldMergeScores() {
        Long userId = 1L;
        int limit = 10;

        var scoring = new RecommendationAlgorithmProperties.Scoring();
        scoring.setCollaborativeWeight(1.0);
        scoring.setContentWeight(0.0);
        scoring.setAiModelWeight(0.0);
        scoring.setPopularityWeight(0.0);

        when(algorithmProperties.getScoring()).thenReturn(scoring);
        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);

        RecommendationResult userCfRec = new RecommendationResult();
        userCfRec.setUserId(userId); userCfRec.setProductId(100L); userCfRec.setScore(50.0);
        userCfRec.setAlgorithm(RecommendationResult.RecommendationAlgorithm.USER_BASED_CF);
        when(userBasedCFService.userBasedCollaborativeFiltering(userId, limit * 2))
            .thenReturn(List.of(userCfRec));
        when(itemBasedCFService.itemBasedCollaborativeFiltering(userId, limit * 2)).thenReturn(List.of());
        when(contentBasedService.contentBasedFiltering(userId, limit * 2)).thenReturn(List.of());
        when(popularityService.popularityBasedRecommendation(userId, limit * 2)).thenReturn(List.of());

        List<RecommendationResult> results = service.hybridRecommendation(userId, limit);

        assertFalse(results.isEmpty());
        assertEquals(100L, results.get(0).getProductId());
        assertEquals(50.0, results.get(0).getScore(), 0.001);
    }
}
