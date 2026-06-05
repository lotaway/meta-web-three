package com.metawebthree.recommendation.infrastructure.config;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class RecommendationAlgorithmPropertiesTest {

    @Test
    void defaultValues_shouldBeSet() {
        RecommendationAlgorithmProperties props = new RecommendationAlgorithmProperties();

        assertEquals(0.4, props.getScoring().getCollaborativeWeight());
        assertEquals(0.3, props.getScoring().getContentWeight());
        assertEquals(0.5, props.getScoring().getHybridWeight());
        assertEquals(0.6, props.getScoring().getPopularityWeight());
        assertEquals(0.8, props.getScoring().getAiModelWeight());
        assertEquals(80.0, props.getScoring().getBaseScore());
        assertEquals(5.0, props.getScoring().getScoreDecay());
        assertEquals(3.5, props.getCtr().getIndustryAverage());
        assertEquals(1.2, props.getConversion().getIndustryAverage());
        assertEquals(20, props.getSimilarUserMaxCount());
        assertEquals(7, props.getRecommendationExpiryDays());
        assertEquals(100.0, props.getPopularityBaseScore());
    }

    @Test
    void setters_shouldUpdateValues() {
        RecommendationAlgorithmProperties props = new RecommendationAlgorithmProperties();

        props.getScoring().setCollaborativeWeight(0.5);
        props.getCtr().setIndustryAverage(4.0);
        props.setSimilarUserMaxCount(50);

        assertEquals(0.5, props.getScoring().getCollaborativeWeight());
        assertEquals(4.0, props.getCtr().getIndustryAverage());
        assertEquals(50, props.getSimilarUserMaxCount());
    }

    @Test
    void behaviorWeights_shouldHaveDefaultValues() {
        var behavior = new RecommendationAlgorithmProperties.Behavior();

        assertEquals(5.0, behavior.getPurchaseWeight());
        assertEquals(4.0, behavior.getCartWeight());
        assertEquals(3.0, behavior.getCollectWeight());
        assertEquals(2.0, behavior.getClickWeight());
        assertEquals(1.0, behavior.getViewWeight());
        assertEquals(1.0, behavior.getDefaultWeight());
    }
}
