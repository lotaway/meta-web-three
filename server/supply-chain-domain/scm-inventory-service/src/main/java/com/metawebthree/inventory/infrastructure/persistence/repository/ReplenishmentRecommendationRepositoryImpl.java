package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.ReplenishmentRecommendation;
import com.metawebthree.inventory.infrastructure.persistence.converter.ReplenishmentRecommendationConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.ReplenishmentRecommendationDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.ReplenishmentRecommendationMapper;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ReplenishmentRecommendationRepositoryImpl implements ReplenishmentRecommendationRepository {

    private final ReplenishmentRecommendationMapper recommendationMapper;
    private final ReplenishmentRecommendationConverter converter;

    public ReplenishmentRecommendationRepositoryImpl(
            ReplenishmentRecommendationMapper recommendationMapper,
            ReplenishmentRecommendationConverter converter) {
        this.recommendationMapper = recommendationMapper;
        this.converter = converter;
    }

    @Override
    public Optional<ReplenishmentRecommendation> findById(Long id) {
        ReplenishmentRecommendationDO entity = recommendationMapper.selectById(id);
        return Optional.ofNullable(converter.toEntity(entity));
    }

    @Override
    public List<ReplenishmentRecommendation> findByWarehouse(Long warehouseId) {
        LambdaQueryWrapper<ReplenishmentRecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReplenishmentRecommendationDO::getWarehouseId, warehouseId);
        return recommendationMapper.selectList(wrapper).stream()
                .map(converter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<ReplenishmentRecommendation> findByStatus(String status) {
        LambdaQueryWrapper<ReplenishmentRecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReplenishmentRecommendationDO::getStatus, status);
        return recommendationMapper.selectList(wrapper).stream()
                .map(converter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<ReplenishmentRecommendation> findPendingRecommendations() {
        LambdaQueryWrapper<ReplenishmentRecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ReplenishmentRecommendationDO::getStatus, "PENDING");
        return recommendationMapper.selectList(wrapper).stream()
                .map(converter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public ReplenishmentRecommendation save(ReplenishmentRecommendation recommendation) {
        ReplenishmentRecommendationDO entityDO = converter.toDto(recommendation);
        if (recommendation.getId() == null) {
            entityDO.setCreatedAt(LocalDateTime.now());
            entityDO.setUpdatedAt(LocalDateTime.now());
            recommendationMapper.insert(entityDO);
            recommendation.setId(entityDO.getId());
        } else {
            entityDO.setUpdatedAt(LocalDateTime.now());
            recommendationMapper.updateById(entityDO);
        }
        return recommendation;
    }

    @Override
    public void delete(ReplenishmentRecommendation recommendation) {
        if (recommendation.getId() != null) {
            recommendationMapper.deleteById(recommendation.getId());
        }
    }
}