package com.metawebthree.recommendation.application.command;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.domain.service.RecommendationDomainService;
import com.metawebthree.recommendation.infrastructure.event.RecommendationEventPublisher;

import java.time.LocalDateTime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class RecommendationCommandServiceTest {

    @Mock
    private RecommendationDomainService domainService;
    @Mock
    private RecommendationEventPublisher eventPublisher;
    @Mock
    private UserBehaviorRepository userBehaviorRepository;
    @Mock
    private RecommendationResultRepository recommendationResultRepository;
    @Mock
    private RecommendationRuleRepository ruleRepository;

    @InjectMocks
    private RecommendationCommandService commandService;

    @Test
    void generateRecommendation_shouldPublishEvent() {
        Long userId = 1L;
        String scene = "home";
        var algorithm = Recommendation.RecommendationAlgorithm.HYBRID;
        int maxItems = 10;

        Recommendation recommendation = new Recommendation();
        recommendation.setId(100L);
        recommendation.setUserId(userId);
        recommendation.setScene(scene);

        when(domainService.generateRecommendation(userId, scene, algorithm, maxItems))
            .thenReturn(recommendation);

        Recommendation result = commandService.generateRecommendation(userId, scene, algorithm, maxItems);

        assertEquals(100L, result.getId());
        verify(eventPublisher).publishRecommendationGenerated(100L, userId, scene);
    }

    @Test
    void recordBehavior_shouldPublishEvent() {
        commandService.recordBehavior(1L, "SKU001", "VIEW");

        verify(domainService).recordUserBehavior(1L, "SKU001", "VIEW");
        verify(eventPublisher).publishUserBehaviorRecorded(1L, "SKU001", "VIEW");
    }

    @Test
    void createRule_shouldPublishEvent() {
        var rule = new RecommendationRule();
        rule.setId(1L);
        rule.setRuleName("Test Rule");
        rule.setScene("home");

        when(domainService.createRule("Test Rule", "home", RecommendationRule.RuleType.BOOST))
            .thenReturn(rule);

        RecommendationRule result = commandService.createRule("Test Rule", "home", RecommendationRule.RuleType.BOOST);

        assertEquals("Test Rule", result.getRuleName());
        verify(eventPublisher).publishRuleCreated(1L, "Test Rule", "home");
    }

    @Test
    void activateRule_shouldPublishEvent() {
        commandService.activateRule(1L);

        verify(domainService).activateRule(1L);
        verify(eventPublisher).publishRuleActivated(1L);
    }

    @Test
    void markRecommendationClicked_shouldDelegate() {
        commandService.markRecommendationClicked(1L);

        verify(recommendationResultRepository).markAsClicked(1L);
    }

    @Test
    void markRecommendationPurchased_shouldDelegate() {
        commandService.markRecommendationPurchased(1L);

        verify(recommendationResultRepository).markAsPurchased(1L);
    }

    @Test
    void recordUserBehavior_shouldSetCorrectFields() {
        UserBehavior saved = new UserBehavior();
        saved.setId(1L);
        saved.setUserId(1L);
        saved.setProductId(100L);
        saved.setBehaviorType(UserBehavior.BehaviorType.VIEW);
        when(userBehaviorRepository.save(any())).thenReturn(saved);

        UserBehavior result = commandService.recordUserBehavior(
            1L, 100L, UserBehavior.BehaviorType.VIEW, null, "session-1", "home");

        assertEquals(1L, result.getUserId());
        assertEquals(100L, result.getProductId());
        assertEquals(UserBehavior.BehaviorType.VIEW, result.getBehaviorType());
        assertNotNull(result.getTimestamp());
    }

    @Test
    void cleanupOldData_shouldDeleteOldRecords() {
        commandService.cleanupOldData(90);

        verify(userBehaviorRepository).deleteByTimestampBefore(any(LocalDateTime.class));
        verify(recommendationResultRepository).deleteByExpiresAtBefore(any(LocalDateTime.class));
    }
}
