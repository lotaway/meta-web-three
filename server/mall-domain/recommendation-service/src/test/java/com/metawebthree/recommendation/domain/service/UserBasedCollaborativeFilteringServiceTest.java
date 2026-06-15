package com.metawebthree.recommendation.domain.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class UserBasedCollaborativeFilteringServiceTest {

    @Mock
    private UserBehaviorRepository userBehaviorRepository;

    @Mock
    private RecommendationAlgorithmProperties algorithmProperties;

    @InjectMocks
    private UserBasedCollaborativeFilteringService service;

    @Test
    void userBasedCollaborativeFiltering_withSimilarUsers_shouldReturnRecommendations() {
        Long userId = 1L;
        int limit = 5;

        var behavior = new RecommendationAlgorithmProperties.Behavior();
        behavior.setPurchaseWeight(5.0);
        behavior.setViewWeight(1.0);
        var ctr = new RecommendationAlgorithmProperties.CTR();
        ctr.setIndustryAverage(3.5);
        var scoring = new RecommendationAlgorithmProperties.Scoring();
        scoring.setBaseScore(80.0);
        scoring.setScoreDecay(5.0);
        var conv = new RecommendationAlgorithmProperties.Conversion();
        conv.setIndustryAverage(1.2);

        when(algorithmProperties.getSimilarUserMaxCount()).thenReturn(10);
        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(algorithmProperties.getBehavior()).thenReturn(behavior);
        when(algorithmProperties.getCtr()).thenReturn(ctr);
        when(algorithmProperties.getScoring()).thenReturn(scoring);
        when(algorithmProperties.getConversion()).thenReturn(conv);

        UserBehavior b1 = new UserBehavior();
        b1.setUserId(2L); b1.setProductId(100L); b1.setBehaviorType(UserBehavior.BehaviorType.PURCHASE); b1.setBehaviorValue(5.0);
        UserBehavior b2 = new UserBehavior();
        b2.setUserId(2L); b2.setProductId(101L); b2.setBehaviorType(UserBehavior.BehaviorType.PURCHASE); b2.setBehaviorValue(5.0);
        UserBehavior b3 = new UserBehavior();
        b3.setUserId(2L); b3.setProductId(100L); b3.setBehaviorType(UserBehavior.BehaviorType.VIEW); b3.setBehaviorValue(1.0);

        UserBehavior selfB1 = new UserBehavior();
        selfB1.setUserId(1L); selfB1.setProductId(100L); selfB1.setBehaviorType(UserBehavior.BehaviorType.PURCHASE); selfB1.setBehaviorValue(5.0);

        when(userBehaviorRepository.findAll()).thenReturn(List.of(b1, b2, b3, selfB1));
        when(userBehaviorRepository.findByUserIdOrderByTimestampDesc(1L)).thenReturn(List.of(selfB1));
        when(userBehaviorRepository.findByUserIdOrderByTimestampDesc(2L)).thenReturn(List.of(b1, b2, b3));
        when(userBehaviorRepository.findByUserIdAndBehaviorTypeOrderByTimestampDesc(eq(2L), eq(UserBehavior.BehaviorType.PURCHASE)))
            .thenReturn(List.of(b1, b2));

        List<RecommendationResult> results = service.userBasedCollaborativeFiltering(userId, limit);

        assertNotNull(results);
        assertFalse(results.isEmpty());
        assertTrue(results.size() <= limit);
        for (RecommendationResult r : results) {
            assertEquals(userId, r.getUserId());
            assertEquals(RecommendationResult.RecommendationAlgorithm.USER_BASED_CF, r.getAlgorithm());
        }
    }

    @Test
    void calculateUserSimilarity_withSameUser_shouldReturnOne() {
        UserBehavior b1 = new UserBehavior();
        b1.setUserId(1L); b1.setProductId(100L); b1.setBehaviorType(UserBehavior.BehaviorType.PURCHASE); b1.setBehaviorValue(5.0);

        var behavior = new RecommendationAlgorithmProperties.Behavior();
        behavior.setPurchaseWeight(5.0);

        when(algorithmProperties.getBehavior()).thenReturn(behavior);
        when(userBehaviorRepository.findByUserIdOrderByTimestampDesc(1L)).thenReturn(List.of(b1, b1));

        double similarity = service.calculateUserSimilarity(1L, 1L);

        assertEquals(1.0, similarity, 0.001);
    }

    @Test
    void userBasedCollaborativeFiltering_withNoSimilarUsers_shouldReturnEmpty() {
        Long userId = 999L;
        int limit = 5;

        var behavior = new RecommendationAlgorithmProperties.Behavior();
        var ctr = new RecommendationAlgorithmProperties.CTR();
        ctr.setIndustryAverage(3.5);
        var scoring = new RecommendationAlgorithmProperties.Scoring();
        scoring.setBaseScore(80.0);
        scoring.setScoreDecay(5.0);
        var conv = new RecommendationAlgorithmProperties.Conversion();
        conv.setIndustryAverage(1.2);

        when(algorithmProperties.getSimilarUserMaxCount()).thenReturn(10);
        when(algorithmProperties.getRecommendationExpiryDays()).thenReturn(7);
        when(algorithmProperties.getBehavior()).thenReturn(behavior);
        when(algorithmProperties.getCtr()).thenReturn(ctr);
        when(algorithmProperties.getScoring()).thenReturn(scoring);
        when(algorithmProperties.getConversion()).thenReturn(conv);
        when(userBehaviorRepository.findAll()).thenReturn(List.of());

        List<RecommendationResult> results = service.userBasedCollaborativeFiltering(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }
}
