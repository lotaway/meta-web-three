package com.metawebthree.product_recommendation.application.service;

import com.metawebthree.product_recommendation.application.dto.ProductProfileDTO;
import com.metawebthree.product_recommendation.application.dto.RecommendationDTO;
import com.metawebthree.product_recommendation.application.dto.UserBehaviorDTO;
import com.metawebthree.product_recommendation.domain.model.ProductProfile;
import com.metawebthree.product_recommendation.domain.model.Recommendation;
import com.metawebthree.product_recommendation.domain.model.RecommendationType;
import com.metawebthree.product_recommendation.domain.model.UserBehavior;
import com.metawebthree.product_recommendation.domain.repository.ProductProfileRepository;
import com.metawebthree.product_recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.product_recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.product_recommendation.domain.service.RecommendationDomainService;
import com.metawebthree.product_recommendation.infrastructure.rpc.ProductServiceClient;
import com.metawebthree.product_recommendation.infrastructure.rpc.UserBehaviorServiceClient;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

public class RecommendationApplicationService {

    private final RecommendationDomainService recommendationDomainService;
    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductProfileRepository productProfileRepository;
    private final ProductServiceClient productServiceClient;
    private final UserBehaviorServiceClient userBehaviorServiceClient;

    public RecommendationApplicationService(
            RecommendationDomainService recommendationDomainService,
            UserBehaviorRepository userBehaviorRepository,
            ProductProfileRepository productProfileRepository,
            ProductServiceClient productServiceClient,
            UserBehaviorServiceClient userBehaviorServiceClient) {
        this.recommendationDomainService = recommendationDomainService;
        this.userBehaviorRepository = userBehaviorRepository;
        this.productProfileRepository = productProfileRepository;
        this.productServiceClient = productServiceClient;
        this.userBehaviorServiceClient = userBehaviorServiceClient;
    }

    public List<RecommendationDTO> getRecommendations(Long userId, String type, int limit) {
        List<Recommendation> recommendations;

        if (type == null || type.isEmpty()) {
            recommendations = recommendationDomainService.generateHybridRecommendations(userId, limit);
        } else if ("COLLABORATIVE_FILTERING".equalsIgnoreCase(type)) {
            recommendations = recommendationDomainService.generateCollaborativeFilteringRecommendations(userId, limit);
        } else if ("CONTENT_BASED".equalsIgnoreCase(type)) {
            recommendations = recommendationDomainService.generateContentBasedRecommendations(userId, limit);
        } else {
            recommendations = recommendationDomainService.generateHybridRecommendations(userId, limit);
        }

        recommendationDomainService.saveRecommendations(recommendations);

        return recommendations.stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    public void recordUserBehavior(UserBehaviorDTO dto) {
        UserBehavior behavior = new UserBehavior();
        behavior.setUserId(dto.getUserId());
        behavior.setProductId(dto.getProductId());
        behavior.setBehaviorType(UserBehavior.BehaviorType.valueOf(dto.getBehaviorType()));
        behavior.setDurationSeconds(dto.getDurationSeconds());
        behavior.setSource(dto.getSource());
        behavior.setSearchKeyword(dto.getSearchKeyword());
        behavior.setOccurredAt(dto.getOccurredAt() != null ? dto.getOccurredAt() : LocalDateTime.now());

        userBehaviorRepository.save(behavior);
    }

    public void syncUserBehaviorsFromExternal(Long userId) {
        List<UserBehaviorDTO> externalBehaviors = userBehaviorServiceClient.getUserBehaviors(userId);
        List<UserBehavior> behaviors = externalBehaviors.stream()
                .map(dto -> {
                    UserBehavior behavior = new UserBehavior();
                    behavior.setUserId(dto.getUserId());
                    behavior.setProductId(dto.getProductId());
                    behavior.setBehaviorType(UserBehavior.BehaviorType.valueOf(dto.getBehaviorType()));
                    behavior.setDurationSeconds(dto.getDurationSeconds());
                    behavior.setSource(dto.getSource());
                    behavior.setSearchKeyword(dto.getSearchKeyword());
                    behavior.setOccurredAt(dto.getOccurredAt());
                    return behavior;
                })
                .collect(Collectors.toList());

        userBehaviorRepository.batchSave(behaviors);
    }

    public void syncProductProfilesFromExternal() {
        List<ProductProfileDTO> externalProfiles = productServiceClient.getAllProductProfiles();
        List<ProductProfile> profiles = externalProfiles.stream()
                .map(dto -> {
                    ProductProfile profile = new ProductProfile(dto.getProductId(), dto.getCategory());
                    profile.setTags(dto.getTags());
                    profile.setAttributes(dto.getAttributes());
                    profile.setPrice(dto.getPrice());
                    profile.setAverageRating(dto.getAverageRating());
                    profile.setSalesCount(dto.getSalesCount());
                    profile.setEmbedding(dto.getEmbedding());
                    profile.setSimilarProductIds(dto.getSimilarProductIds());
                    return profile;
                })
                .collect(Collectors.toList());

        productProfileRepository.batchSave(profiles);
    }

    public void updateProductProfile(ProductProfileDTO dto) {
        ProductProfile profile = new ProductProfile(dto.getProductId(), dto.getCategory());
        profile.setId(dto.getId());
        profile.setTags(dto.getTags());
        profile.setAttributes(dto.getAttributes());
        profile.setPrice(dto.getPrice());
        profile.setAverageRating(dto.getAverageRating());
        profile.setSalesCount(dto.getSalesCount());
        profile.setEmbedding(dto.getEmbedding());
        profile.setSimilarProductIds(dto.getSimilarProductIds());

        productProfileRepository.save(profile);
    }

    public void cleanupExpiredData() {
        recommendationDomainService.cleanupExpiredRecommendations();
        userBehaviorRepository.deleteOldRecords(LocalDateTime.now().minusDays(90));
    }

    private RecommendationDTO toDTO(Recommendation recommendation) {
        RecommendationDTO dto = new RecommendationDTO();
        dto.setId(recommendation.getId());
        dto.setUserId(recommendation.getUserId());
        dto.setProductId(recommendation.getProductId());
        dto.setScore(recommendation.getScore());
        dto.setType(recommendation.getType() != null ? recommendation.getType().name() : null);
        dto.setReason(recommendation.getReason());
        dto.setCreatedAt(recommendation.getCreatedAt());
        return dto;
    }
}