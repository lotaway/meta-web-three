package com.metawebthree.product_recommendation.infrastructure.persistence.mapper;

import com.metawebthree.product_recommendation.domain.model.UserBehavior;
import com.metawebthree.product_recommendation.infrastructure.persistence.entity.UserBehaviorEntity;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.List;

public class UserBehaviorMapper {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static UserBehaviorEntity toEntity(UserBehavior model) {
        if (model == null) {
            return null;
        }

        UserBehaviorEntity entity = new UserBehaviorEntity();
        entity.setId(model.getId());
        entity.setUserId(model.getUserId());
        entity.setProductId(model.getProductId());
        entity.setBehaviorType(model.getBehaviorType() != null ? model.getBehaviorType().name() : null);
        entity.setDurationSeconds(model.getDurationSeconds());
        entity.setSource(model.getSource());
        entity.setSearchKeyword(model.getSearchKeyword());
        entity.setOccurredAt(model.getOccurredAt());

        return entity;
    }

    public static UserBehavior toModel(UserBehaviorEntity entity) {
        if (entity == null) {
            return null;
        }

        UserBehavior model = new UserBehavior();
        model.setId(entity.getId());
        model.setUserId(entity.getUserId());
        model.setProductId(entity.getProductId());
        model.setBehaviorType(entity.getBehaviorType() != null ? 
                UserBehavior.BehaviorType.valueOf(entity.getBehaviorType()) : null);
        model.setDurationSeconds(entity.getDurationSeconds());
        model.setSource(entity.getSource());
        model.setSearchKeyword(entity.getSearchKeyword());
        model.setOccurredAt(entity.getOccurredAt());

        return model;
    }

    public static List<UserBehavior> toModelList(List<UserBehaviorEntity> entities) {
        if (entities == null) {
            return List.of();
        }
        return entities.stream()
                .map(UserBehaviorMapper::toModel)
                .collect(java.util.stream.Collectors.toList());
    }
}