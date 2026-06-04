package com.metawebthree.recommendation.domain.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.Recommendation.RecommendedItem;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class RecommendationDomainServiceTest {

    @Mock
    private RecommendationRepository recommendationRepository;
    @Mock
    private RecommendationRuleRepository ruleRepository;
    @Mock
    private RecommendationAlgorithmProperties algorithmProperties;
    @Mock
    private UserBehaviorRepository userBehaviorRepository;
    @Mock
    private ProductSimilarityRepository productSimilarityRepository;
    @Mock
    private RecommendationResultRepository recommendationResultRepository;

    @InjectMocks
    private RecommendationDomainServiceImpl service;

    @Captor
    private ArgumentCaptor<Recommendation> recommendationCaptor;
    @Captor
    private ArgumentCaptor<RecommendationRule> ruleCaptor;

    @Test
    void generateRecommendation_shouldCreateWithCorrectFields() {
        Long userId = 1L;
        String scene = "home";
        var algorithm = Recommendation.RecommendationAlgorithm.HYBRID;
        int maxItems = 10;

        RecommendationResult result = new RecommendationResult();
        result.setUserId(userId);
        result.setProductId(100L);
        result.setScore(85.0);
        result.setReason("test reason");
        result.setAlgorithm(RecommendationResult.RecommendationAlgorithm.HYBRID);

        when(recommendationResultRepository.findByUserIdAndAlgorithm(
                eq(userId), eq(RecommendationResult.RecommendationAlgorithm.HYBRID), any(LocalDateTime.class)))
            .thenReturn(List.of(result));
        when(recommendationRepository.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        Recommendation saved = service.generateRecommendation(userId, scene, algorithm, maxItems);

        verify(recommendationRepository).save(recommendationCaptor.capture());
        Recommendation captured = recommendationCaptor.getValue();

        assertEquals(userId, captured.getUserId());
        assertEquals(scene, captured.getScene());
        assertEquals(algorithm, captured.getAlgorithm());
        assertEquals(0, captured.getImpressionCount());
        assertEquals(0, captured.getClickCount());
        assertEquals(0, captured.getConversionCount());
        assertNotNull(captured.getCreatedAt());
        assertNotNull(captured.getExpiresAt());
        assertNotNull(captured.getItems());
        assertEquals(1, captured.getItems().size());
        assertEquals("100", captured.getItems().get(0).getSkuCode());
        assertEquals(0, BigDecimal.valueOf(85.0).compareTo(captured.getItems().get(0).getScore()));
        assertEquals(Recommendation.RecommendationStatus.COMPLETED, captured.getStatus());
    }

    @Test
    void calculateCTR_withValidData_shouldReturnCorrectRate() {
        Long recommendationId = 1L;
        Recommendation recommendation = new Recommendation();
        recommendation.setId(recommendationId);
        recommendation.setImpressionCount(100);
        recommendation.setClickCount(10);

        var ctrProps = new RecommendationAlgorithmProperties.CTR();
        ctrProps.setHighThreshold(100.0);
        ctrProps.setLowThreshold(0.0);

        when(recommendationRepository.findById(recommendationId)).thenReturn(Optional.of(recommendation));
        when(algorithmProperties.getCtr()).thenReturn(ctrProps);

        Double result = service.calculateCTR(recommendationId);

        assertEquals(10.0, result, 0.001);
    }

    @Test
    void calculateCTR_withZeroImpressions_shouldReturnIndustryAverage() {
        Long recommendationId = 1L;
        Recommendation recommendation = new Recommendation();
        recommendation.setId(recommendationId);
        recommendation.setImpressionCount(0);

        var ctrProps = new RecommendationAlgorithmProperties.CTR();
        ctrProps.setIndustryAverage(3.5);

        when(recommendationRepository.findById(recommendationId)).thenReturn(Optional.of(recommendation));
        when(algorithmProperties.getCtr()).thenReturn(ctrProps);

        Double result = service.calculateCTR(recommendationId);

        assertEquals(3.5, result, 0.001);
    }

    @Test
    void calculateConversionRate_withValidData_shouldReturnCorrectRate() {
        Long recommendationId = 1L;
        Recommendation recommendation = new Recommendation();
        recommendation.setId(recommendationId);
        recommendation.setClickCount(200);
        recommendation.setConversionCount(20);

        var convProps = new RecommendationAlgorithmProperties.Conversion();
        convProps.setHighThreshold(100.0);
        convProps.setLowThreshold(0.0);

        when(recommendationRepository.findById(recommendationId)).thenReturn(Optional.of(recommendation));
        when(algorithmProperties.getConversion()).thenReturn(convProps);

        Double result = service.calculateConversionRate(recommendationId);

        assertEquals(10.0, result, 0.001);
    }

    @Test
    void calculateConversionRate_withZeroClicks_shouldReturnIndustryAverage() {
        Long recommendationId = 1L;
        Recommendation recommendation = new Recommendation();
        recommendation.setId(recommendationId);
        recommendation.setClickCount(0);

        var convProps = new RecommendationAlgorithmProperties.Conversion();
        convProps.setIndustryAverage(1.2);

        when(recommendationRepository.findById(recommendationId)).thenReturn(Optional.of(recommendation));
        when(algorithmProperties.getConversion()).thenReturn(convProps);

        Double result = service.calculateConversionRate(recommendationId);

        assertEquals(1.2, result, 0.001);
    }

    @Test
    void activateRule_shouldActivate() {
        Long ruleId = 1L;
        RecommendationRule rule = new RecommendationRule();
        rule.setId(ruleId);
        rule.setStatus(RecommendationRule.RuleStatus.DRAFT);

        when(ruleRepository.findById(ruleId)).thenReturn(Optional.of(rule));

        service.activateRule(ruleId);

        verify(ruleRepository).update(ruleCaptor.capture());
        assertEquals(RecommendationRule.RuleStatus.ACTIVE, ruleCaptor.getValue().getStatus());
    }

    @Test
    void activateRule_whenAlreadyActive_shouldRemainActive() {
        Long ruleId = 1L;
        RecommendationRule rule = new RecommendationRule();
        rule.setId(ruleId);
        rule.setStatus(RecommendationRule.RuleStatus.ACTIVE);

        when(ruleRepository.findById(ruleId)).thenReturn(Optional.of(rule));

        service.activateRule(ruleId);

        verify(ruleRepository).update(ruleCaptor.capture());
        assertEquals(RecommendationRule.RuleStatus.ACTIVE, ruleCaptor.getValue().getStatus());
    }

    @Test
    void activateRule_whenRuleNotFound_shouldThrow() {
        when(ruleRepository.findById(anyLong())).thenReturn(Optional.empty());

        assertThrows(IllegalArgumentException.class, () -> service.activateRule(999L));
    }

    @Test
    void createRule_shouldCreateWithCorrectFields() {
        String ruleName = "Test Rule";
        String scene = "home";
        var type = RecommendationRule.RuleType.BOOST;

        when(ruleRepository.save(any())).thenAnswer(invocation -> invocation.getArgument(0));

        RecommendationRule created = service.createRule(ruleName, scene, type);

        verify(ruleRepository).save(ruleCaptor.capture());
        RecommendationRule captured = ruleCaptor.getValue();

        assertEquals(ruleName, captured.getRuleName());
        assertEquals(scene, captured.getScene());
        assertEquals(type, captured.getType());
        assertEquals(RecommendationRule.RuleStatus.DRAFT, captured.getStatus());
        assertNotNull(captured.getCreatedAt());
        assertNotNull(captured.getUpdatedAt());
    }

    @Test
    void getUserRecommendations_shouldReturnUserRecommendations() {
        Long userId = 1L;
        String scene = "home";
        Recommendation rec1 = new Recommendation();
        rec1.setId(1L);
        rec1.setUserId(userId);
        rec1.setScene(scene);

        when(recommendationRepository.findByUserIdAndScene(userId, scene))
            .thenReturn(List.of(rec1));

        List<Recommendation> result = service.getUserRecommendations(userId, scene);

        assertEquals(1, result.size());
        assertEquals(userId, result.get(0).getUserId());
        assertEquals(scene, result.get(0).getScene());
    }
}
