package com.metawebthree.product_recommendation.interfaces.controller;

import com.metawebthree.product_recommendation.application.dto.ProductProfileDTO;
import com.metawebthree.product_recommendation.application.dto.RecommendationDTO;
import com.metawebthree.product_recommendation.application.dto.UserBehaviorDTO;
import com.metawebthree.product_recommendation.application.service.RecommendationApplicationService;

import java.util.List;
import java.util.Map;

public class ProductRecommendationController {

    private final RecommendationApplicationService recommendationApplicationService;

    public ProductRecommendationController(RecommendationApplicationService recommendationApplicationService) {
        this.recommendationApplicationService = recommendationApplicationService;
    }

    public List<RecommendationDTO> getRecommendations(Long userId, String type, Integer limit) {
        int actualLimit = limit != null && limit > 0 ? limit : 10;
        return recommendationApplicationService.getRecommendations(userId, type, actualLimit);
    }

    public void recordBehavior(UserBehaviorDTO behavior) {
        recommendationApplicationService.recordUserBehavior(behavior);
    }

    public void updateProductProfile(ProductProfileDTO profile) {
        recommendationApplicationService.updateProductProfile(profile);
    }

    public void syncProductProfiles() {
        recommendationApplicationService.syncProductProfilesFromExternal();
    }

    public void cleanupExpiredData() {
        recommendationApplicationService.cleanupExpiredData();
    }
}