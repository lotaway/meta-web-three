package com.metawebthree.inventory.infrastructure.persistence.converter;

import com.metawebthree.inventory.domain.entity.ReplenishmentRecommendation;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.ReplenishmentRecommendationDO;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;

@Component
public class ReplenishmentRecommendationConverter {

    public ReplenishmentRecommendation toEntity(ReplenishmentRecommendationDO dto) {
        if (dto == null) {
            return null;
        }
        ReplenishmentRecommendation entity = new ReplenishmentRecommendation();
        entity.setId(dto.getId());
        entity.setSkuCode(dto.getSkuCode());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setCurrentStock(dto.getCurrentStock());
        entity.setSafetyStock(dto.getSafetyStock());
        entity.setLeadTimeDays(dto.getLeadTimeDays());
        entity.setAverageDailySales(dto.getAverageDailySales());
        entity.setRecommendedQuantity(dto.getRecommendedQuantity());
        entity.setRecommendationType(dto.getRecommendationType());
        entity.setStatus(dto.getStatus());
        entity.setGeneratedAt(dto.getGeneratedAt());
        entity.setCreatedAt(dto.getCreatedAt());
        entity.setUpdatedAt(dto.getUpdatedAt());
        return entity;
    }

    public ReplenishmentRecommendationDO toDto(ReplenishmentRecommendation entity) {
        if (entity == null) {
            return null;
        }
        ReplenishmentRecommendationDO dto = new ReplenishmentRecommendationDO();
        dto.setId(entity.getId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setCurrentStock(entity.getCurrentStock());
        dto.setSafetyStock(entity.getSafetyStock());
        dto.setLeadTimeDays(entity.getLeadTimeDays());
        dto.setAverageDailySales(entity.getAverageDailySales());
        dto.setRecommendedQuantity(entity.getRecommendedQuantity());
        dto.setRecommendationType(entity.getRecommendationType());
        dto.setStatus(entity.getStatus());
        dto.setGeneratedAt(entity.getGeneratedAt());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
    }

    public List<ReplenishmentRecommendation> toEntityList(List<ReplenishmentRecommendationDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        List<ReplenishmentRecommendation> list = new ArrayList<>();
        for (ReplenishmentRecommendationDO dto : dtoList) {
            list.add(toEntity(dto));
        }
        return list;
    }
}