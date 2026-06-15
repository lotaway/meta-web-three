package com.metawebthree.recommendation.domain.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class PopularityBasedRecommendationServiceTest {

    @Mock
    private RecommendationResultRepository recommendationResultRepository;

    @Mock
    private RecommendationAlgorithmProperties algorithmProperties;

    @InjectMocks
    private PopularityBasedRecommendationService service;

    @Test
    void popularityBasedRecommendation_withResults_shouldReturnSortedLimited() {
        Long userId = 1L;
        int limit = 2;

        RecommendationResult r1 = new RecommendationResult();
        r1.setUserId(userId); r1.setProductId(100L); r1.setScore(90.0);
        r1.setAlgorithm(RecommendationResult.RecommendationAlgorithm.POPULARITY);

        RecommendationResult r2 = new RecommendationResult();
        r2.setUserId(userId); r2.setProductId(200L); r2.setScore(80.0);
        r2.setAlgorithm(RecommendationResult.RecommendationAlgorithm.POPULARITY);

        RecommendationResult r3 = new RecommendationResult();
        r3.setUserId(userId); r3.setProductId(300L); r3.setScore(70.0);
        r3.setAlgorithm(RecommendationResult.RecommendationAlgorithm.POPULARITY);

        when(recommendationResultRepository.findByUserIdAndAlgorithm(
            eq(userId), eq(RecommendationResult.RecommendationAlgorithm.POPULARITY), any(LocalDateTime.class)))
            .thenReturn(List.of(r1, r2, r3));

        List<RecommendationResult> results = service.popularityBasedRecommendation(userId, limit);

        assertNotNull(results);
        assertEquals(limit, results.size());
        assertEquals(100L, results.get(0).getProductId());
        assertEquals(200L, results.get(1).getProductId());
    }

    @Test
    void popularityBasedRecommendation_withNoResults_shouldReturnEmpty() {
        Long userId = 1L;
        int limit = 10;

        when(recommendationResultRepository.findByUserIdAndAlgorithm(
            eq(userId), eq(RecommendationResult.RecommendationAlgorithm.POPULARITY), any(LocalDateTime.class)))
            .thenReturn(List.of());

        List<RecommendationResult> results = service.popularityBasedRecommendation(userId, limit);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }

    @Test
    void popularityBasedRecommendation_limitExceedsResults_shouldReturnAll() {
        Long userId = 1L;
        int limit = 10;

        RecommendationResult r1 = new RecommendationResult();
        r1.setUserId(userId); r1.setProductId(100L); r1.setScore(85.0);
        r1.setAlgorithm(RecommendationResult.RecommendationAlgorithm.POPULARITY);

        when(recommendationResultRepository.findByUserIdAndAlgorithm(
            eq(userId), eq(RecommendationResult.RecommendationAlgorithm.POPULARITY), any(LocalDateTime.class)))
            .thenReturn(List.of(r1));

        List<RecommendationResult> results = service.popularityBasedRecommendation(userId, limit);

        assertEquals(1, results.size());
        assertEquals(100L, results.get(0).getProductId());
    }
}
