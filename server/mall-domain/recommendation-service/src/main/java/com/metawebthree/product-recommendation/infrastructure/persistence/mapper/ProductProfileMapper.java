package com.metawebthree.product_recommendation.infrastructure.persistence.mapper;

import com.metawebthree.product_recommendation.domain.model.ProductProfile;
import com.metawebthree.product_recommendation.infrastructure.persistence.entity.ProductProfileEntity;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProductProfileMapper {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static ProductProfileEntity toEntity(ProductProfile model) {
        if (model == null) {
            return null;
        }

        ProductProfileEntity entity = new ProductProfileEntity();
        entity.setId(model.getId());
        entity.setProductId(model.getProductId());
        entity.setCategory(model.getCategory());
        entity.setPrice(model.getPrice());
        entity.setAverageRating(model.getAverageRating());
        entity.setSalesCount(model.getSalesCount());

        if (model.getTags() != null) {
            try {
                entity.setTags(objectMapper.writeValueAsString(model.getTags()));
            } catch (JsonProcessingException e) {
                entity.setTags("[]");
            }
        }

        if (model.getAttributes() != null) {
            try {
                entity.setAttributes(objectMapper.writeValueAsString(model.getAttributes()));
            } catch (JsonProcessingException e) {
                entity.setAttributes("[]");
            }
        }

        if (model.getEmbedding() != null) {
            try {
                entity.setEmbedding(objectMapper.writeValueAsString(model.getEmbedding()));
            } catch (JsonProcessingException e) {
                entity.setEmbedding("{}");
            }
        }

        if (model.getSimilarProductIds() != null) {
            try {
                entity.setSimilarProductIds(objectMapper.writeValueAsString(model.getSimilarProductIds()));
            } catch (JsonProcessingException e) {
                entity.setSimilarProductIds("[]");
            }
        }

        return entity;
    }

    public static ProductProfile toModel(ProductProfileEntity entity) {
        if (entity == null) {
            return null;
        }

        ProductProfile model = new ProductProfile();
        model.setId(entity.getId());
        model.setProductId(entity.getProductId());
        model.setCategory(entity.getCategory());
        model.setPrice(entity.getPrice());
        model.setAverageRating(entity.getAverageRating());
        model.setSalesCount(entity.getSalesCount());

        if (entity.getTags() != null && !entity.getTags().isEmpty()) {
            try {
                model.setTags(objectMapper.readValue(entity.getTags(), new TypeReference<List<String>>() {}));
            } catch (JsonProcessingException e) {
                model.setTags(List.of());
            }
        }

        if (entity.getAttributes() != null && !entity.getAttributes().isEmpty()) {
            try {
                model.setAttributes(objectMapper.readValue(entity.getAttributes(), new TypeReference<List<String>>() {}));
            } catch (JsonProcessingException e) {
                model.setAttributes(List.of());
            }
        }

        if (entity.getEmbedding() != null && !entity.getEmbedding().isEmpty()) {
            try {
                model.setEmbedding(objectMapper.readValue(entity.getEmbedding(), new TypeReference<Map<String, Double>>() {}));
            } catch (JsonProcessingException e) {
                model.setEmbedding(new HashMap<>());
            }
        }

        if (entity.getSimilarProductIds() != null && !entity.getSimilarProductIds().isEmpty()) {
            try {
                model.setSimilarProductIds(objectMapper.readValue(entity.getSimilarProductIds(), new TypeReference<List<Long>>() {}));
            } catch (JsonProcessingException e) {
                model.setSimilarProductIds(List.of());
            }
        }

        return model;
    }

    public static List<ProductProfile> toModelList(List<ProductProfileEntity> entities) {
        if (entities == null) {
            return List.of();
        }
        return entities.stream()
                .map(ProductProfileMapper::toModel)
                .collect(java.util.stream.Collectors.toList());
    }
}