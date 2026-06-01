package com.metawebthree.product_recommendation.infrastructure.persistence.mapper;

import com.metawebthree.product_recommendation.domain.model.Recommendation;
import com.metawebthree.product_recommendation.domain.model.RecommendationType;
import com.metawebthree.product_recommendation.infrastructure.persistence.entity.RecommendationEntity;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

public class RecommendationMapper {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static RecommendationEntity toEntity(Recommendation model) {
        if (model == null) {
            return null;
        }

        RecommendationEntity entity = new RecommendationEntity();
        entity.setId(model.getId());
        entity.setUserId(model.getUserId());
        entity.setProductId(model.getProductId());
        entity.setScore(model.getScore());
        entity.setType(model.getType() != null ? model.getType().name() : null);
        entity.setReason(model.getReason());
        entity.setCreatedAt(model.getCreatedAt());
        entity.setExpiresAt(model.getExpiresAt());

        if (model.getFeatures() != null) {
            try {
                entity.setFeatures(objectMapper.writeValueAsString(model.getFeatures()));
            } catch (JsonProcessingException e) {
                entity.setFeatures("{}");
            }
        }

        return entity;
    }

    public static Recommendation toModel(RecommendationEntity entity) {
        if (entity == null) {
            return null;
        }

        Recommendation model = new Recommendation();
        model.setId(entity.getId());
        model.setUserId(entity.getUserId());
        model.setProductId(entity.getProductId());
        model.setScore(entity.getScore());
        model.setType(entity.getType() != null ? RecommendationType.valueOf(entity.getType()) : null);
        model.setReason(entity.getReason());
        model.setCreatedAt(entity.getCreatedAt());
        model.setExpiresAt(entity.getExpiresAt());

        if (entity.getFeatures() != null && !entity.getFeatures().isEmpty()) {
            try {
                Map<String, Object> features = objectMapper.readValue(
                        entity.getFeatures(), 
                        new TypeReference<HashMap<String, Object>>() {});
                model.setFeatures(features);
            } catch (JsonProcessingException e) {
                model.setFeatures(new HashMap<>());
            }
        }

        return model;
    }
}