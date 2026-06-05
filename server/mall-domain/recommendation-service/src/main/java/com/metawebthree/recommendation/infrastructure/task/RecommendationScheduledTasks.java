package com.metawebthree.recommendation.infrastructure.task;

import com.metawebthree.recommendation.application.command.RecommendationCommandService;
import com.metawebthree.recommendation.domain.entity.ProductSimilarity;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.ProductSimilarityRepository;
import com.metawebthree.recommendation.domain.repository.UserBehaviorRepository;
import com.metawebthree.recommendation.domain.service.RecommendationCalculationUtils;
import com.metawebthree.recommendation.infrastructure.config.RecommendationAlgorithmProperties;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Component
public class RecommendationScheduledTasks {

    private final RecommendationCommandService commandService;
    private final UserBehaviorRepository userBehaviorRepository;
    private final ProductSimilarityRepository productSimilarityRepository;
    private final RecommendationAlgorithmProperties algorithmProperties;

    public RecommendationScheduledTasks(
            RecommendationCommandService commandService,
            UserBehaviorRepository userBehaviorRepository,
            ProductSimilarityRepository productSimilarityRepository,
            RecommendationAlgorithmProperties algorithmProperties) {
        this.commandService = commandService;
        this.userBehaviorRepository = userBehaviorRepository;
        this.productSimilarityRepository = productSimilarityRepository;
        this.algorithmProperties = algorithmProperties;
    }

    @Scheduled(cron = "0 0 3 * * ?")
    public void computePopularityRecommendations() {
        log.info("Starting scheduled popularity recommendation computation");
        try {
            List<UserBehavior> allBehaviors = userBehaviorRepository.findAll();
            Map<Long, Double> productPopularity = new HashMap<>();
            for (UserBehavior behavior : allBehaviors) {
                double weight = switch (behavior.getBehaviorType()) {
                    case PURCHASE -> 5.0;
                    case CART -> 4.0;
                    case COLLECT -> 3.0;
                    case CLICK -> 2.0;
                    default -> 1.0;
                };
                productPopularity.merge(behavior.getProductId(), weight, Double::sum);
            }
            List<Long> sortedProducts = productPopularity.entrySet().stream()
                    .sorted(Map.Entry.<Long, Double>comparingByValue().reversed())
                    .limit(100)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());
            LocalDateTime now = LocalDateTime.now();
            int expiryDays = algorithmProperties.getRecommendationExpiryDays();
            for (Long productId : sortedProducts) {
                RecommendationResult result = RecommendationCalculationUtils.createRecommendationResult(
                        0L, productId, productPopularity.get(productId),
                        RecommendationResult.RecommendationAlgorithm.POPULARITY, expiryDays);
                result.setCreatedAt(now);
                result.setExpiresAt(now.plusDays(expiryDays));
            }
            log.info("Popularity recommendation computation completed, processed {} products", sortedProducts.size());
        } catch (Exception e) {
            log.error("Failed to compute popularity recommendations", e);
        }
    }

    @Scheduled(cron = "0 30 3 * * ?")
    public void updateSimilarityMatrix() {
        log.info("Starting scheduled similarity matrix update");
        try {
            List<UserBehavior> allBehaviors = userBehaviorRepository.findAll();
            Set<Long> productIds = allBehaviors.stream()
                    .map(UserBehavior::getProductId)
                    .collect(Collectors.toSet());
            List<Long> productIdList = new ArrayList<>(productIds);
            int count = 0;
            for (int i = 0; i < productIdList.size(); i++) {
                for (int j = i + 1; j < productIdList.size(); j++) {
                    Long pid1 = productIdList.get(i);
                    Long pid2 = productIdList.get(j);
                    if (productSimilarityRepository.existsSimilarity(pid1, pid2)) {
                        continue;
                    }
                    double similarity = RecommendationCalculationUtils.calculateJaccardSimilarity(
                            userBehaviorRepository, pid1, pid2);
                    if (similarity > 0.1) {
                        ProductSimilarity ps = new ProductSimilarity();
                        ps.setProductId1(pid1);
                        ps.setProductId2(pid2);
                        ps.setSimilarityScore(similarity);
                        ps.setAlgorithm(ProductSimilarity.SimilarityAlgorithm.HYBRID);
                        ps.setLastUpdated(LocalDateTime.now());
                        ps.setUpdateCount(1);
                        productSimilarityRepository.save(ps);
                        count++;
                    }
                }
            }
            log.info("Similarity matrix update completed, added {} new similarities", count);
        } catch (Exception e) {
            log.error("Failed to update similarity matrix", e);
        }
    }

    @Scheduled(cron = "0 0 4 * * ?")
    public void cleanupExpiredData() {
        log.info("Starting scheduled data cleanup");
        try {
            commandService.cleanupOldData(90);
            log.info("Data cleanup completed");
        } catch (Exception e) {
            log.error("Failed to cleanup data", e);
        }
    }
}
