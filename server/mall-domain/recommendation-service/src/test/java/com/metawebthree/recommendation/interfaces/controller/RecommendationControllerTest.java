package com.metawebthree.recommendation.interfaces.controller;

import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.recommendation.application.command.RecommendationCommandService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.bean.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(RecommendationController.class)
class RecommendationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private RecommendationCommandService commandService;

    @MockBean
    private RecommendationQueryService queryService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void getRecommendation_shouldReturn200() throws Exception {
        Recommendation rec = new Recommendation();
        rec.setId(1L);

        when(queryService.getRecommendationById(1L)).thenReturn(Optional.of(rec));

        mockMvc.perform(get("/api/recommendation/1"))
            .andExpect(status().isOk());
    }

    @Test
    void getRecommendation_notFound_shouldReturn404() throws Exception {
        when(queryService.getRecommendationById(999L)).thenReturn(Optional.empty());

        mockMvc.perform(get("/api/recommendation/999"))
            .andExpect(status().isNotFound());
    }

    @Test
    void getUserRecommendations_shouldReturn200() throws Exception {
        Recommendation rec = new Recommendation();
        rec.setId(1L);
        rec.setUserId(1L);

        when(queryService.getUserRecommendations(1L)).thenReturn(List.of(rec));

        mockMvc.perform(get("/api/recommendation/user/1"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$[0].id").value(1));
    }

    @Test
    void markRecommendationClicked_shouldReturn200() throws Exception {
        doNothing().when(commandService).markRecommendationClicked(1L);

        mockMvc.perform(post("/api/recommendation/1/click"))
            .andExpect(status().isOk());
    }

    @Test
    void markRecommendationPurchased_shouldReturn200() throws Exception {
        doNothing().when(commandService).markRecommendationPurchased(1L);

        mockMvc.perform(post("/api/recommendation/1/purchase"))
            .andExpect(status().isOk());
    }

    @Test
    void markPurchasedByProduct_shouldReturn200() throws Exception {
        doNothing().when(commandService).markPurchasedByProduct(1L, 100L);

        mockMvc.perform(post("/api/recommendation/purchase-by-product")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("userId", 1, "productId", 100))))
            .andExpect(status().isOk());

        verify(commandService).markPurchasedByProduct(1L, 100L);
    }

    @Test
    void recordBehavior_shouldReturn200() throws Exception {
        doNothing().when(commandService).recordBehavior(1L, "SKU001", "VIEW");

        mockMvc.perform(post("/api/recommendation/behavior")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "userId", 1, "skuCode", "SKU001", "behaviorType", "VIEW"))))
            .andExpect(status().isOk());
    }

    @Test
    void createRule_shouldReturn200() throws Exception {
        RecommendationRule rule = new RecommendationRule();
        rule.setId(1L);
        rule.setRuleName("Test Rule");

        when(commandService.createRule("Test Rule", "home", RecommendationRule.RuleType.BOOST))
            .thenReturn(rule);

        mockMvc.perform(post("/api/recommendation/rule")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "ruleName", "Test Rule", "scene", "home", "type", "BOOST"))))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.id").value(1));
    }

    @Test
    void searchRulesByScene_shouldReturn200() throws Exception {
        RecommendationRule rule = new RecommendationRule();
        rule.setId(1L);
        rule.setScene("home");

        when(queryService.getRulesByScene("home")).thenReturn(List.of(rule));

        mockMvc.perform(get("/api/recommendation/rule/scene/home"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$[0].id").value(1));
    }

    @Test
    void getBehaviorHistory_shouldReturn200() throws Exception {
        UserBehavior behavior = new UserBehavior();
        behavior.setId(1L);
        behavior.setUserId(1L);

        when(queryService.getUserBehaviorHistory(1L, 50)).thenReturn(List.of(behavior));

        mockMvc.perform(get("/api/recommendation/behavior/user/1?limit=50"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$[0].id").value(1));
    }

    @Test
    void generateRecommendation_shouldReturn200() throws Exception {
        Recommendation rec = new Recommendation();
        rec.setId(1L);

        when(commandService.generateRecommendation(1L, "home",
            Recommendation.RecommendationAlgorithm.HYBRID, 10)).thenReturn(rec);

        mockMvc.perform(post("/api/recommendation/generate")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "userId", 1, "scene", "home", "algorithm", "HYBRID", "maxItems", 10))))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.id").value(1));
    }

    @Test
    void updateSimilarityMatrix_shouldReturn200() throws Exception {
        doNothing().when(commandService).updateProductSimilarities();

        mockMvc.perform(post("/api/recommendation/admin/update-similarity"))
            .andExpect(status().isOk());
    }

    @Test
    void cleanupOldData_shouldReturn200() throws Exception {
        doNothing().when(commandService).cleanupOldData(90);

        mockMvc.perform(post("/api/recommendation/admin/cleanup?daysToKeep=90"))
            .andExpect(status().isOk());
    }

    @Test
    void getMetrics_shouldReturn200() throws Exception {
        var metrics = new RecommendationQueryService.RecommendationMetrics();
        metrics.setTotalRecommendations(100L);
        metrics.setClickedCount(10L);

        when(queryService.getRecommendationMetrics(1L)).thenReturn(metrics);

        mockMvc.perform(get("/api/recommendation/metrics/user/1"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.totalRecommendations").value(100));
    }
}
