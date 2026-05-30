package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.QualityStandard;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityStandardDO;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class QualityStandardConverter {
    
    public QualityStandard toEntity(QualityStandardDO dto) {
        if (dto == null) {
            return null;
        }
        QualityStandard entity = new QualityStandard();
        entity.setId(dto.getId());
        entity.setSkuCode(dto.getSkuCode());
        entity.setProductName(dto.getProductName());
        entity.setInspectionType(dto.getInspectionType());
        entity.setInspectionLevel(dto.getInspectionLevel());
        entity.setSampleRate(dto.getSampleRate());
        entity.setCheckItems(dto.getCheckItems());
        entity.setAcceptanceQty(dto.getAcceptanceQty());
        entity.setDefectQtyThreshold(dto.getDefectQtyThreshold());
        entity.setWeightTolerance(dto.getWeightTolerance());
        entity.setDimensionTolerance(dto.getDimensionTolerance());
        entity.setPackagingRequirement(dto.getPackagingRequirement());
        entity.setLabelRequirement(dto.getLabelRequirement());
        entity.setIsActive(dto.getIsActive() != null && dto.getIsActive() == 1);
        entity.setRemark(dto.getRemark());
        entity.setCreator(dto.getCreator());
        entity.setCreateTime(dto.getCreateTime());
        entity.setUpdater(dto.getUpdater());
        entity.setUpdateTime(dto.getUpdateTime());
        entity.setDeleted(dto.getDeleted() != null && dto.getDeleted() == 1);
        return entity;
    }
    
    public QualityStandardDO toDO(QualityStandard entity) {
        if (entity == null) {
            return null;
        }
        QualityStandardDO dto = new QualityStandardDO();
        dto.setId(entity.getId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setProductName(entity.getProductName());
        dto.setInspectionType(entity.getInspectionType());
        dto.setInspectionLevel(entity.getInspectionLevel());
        dto.setSampleRate(entity.getSampleRate());
        dto.setCheckItems(entity.getCheckItems());
        dto.setAcceptanceQty(entity.getAcceptanceQty());
        dto.setDefectQtyThreshold(entity.getDefectQtyThreshold());
        dto.setWeightTolerance(entity.getWeightTolerance());
        dto.setDimensionTolerance(entity.getDimensionTolerance());
        dto.setPackagingRequirement(entity.getPackagingRequirement());
        dto.setLabelRequirement(entity.getLabelRequirement());
        dto.setIsActive(entity.getIsActive() != null && entity.getIsActive() ? 1 : 0);
        dto.setRemark(entity.getRemark());
        dto.setCreator(entity.getCreator());
        dto.setCreateTime(entity.getCreateTime());
        dto.setUpdater(entity.getUpdater());
        dto.setUpdateTime(entity.getUpdateTime());
        dto.setDeleted(entity.getDeleted() != null && entity.getDeleted() ? 1 : 0);
        return dto;
    }
    
    public List<QualityStandard> toEntityList(List<QualityStandardDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        return dtoList.stream()
                .map(this::toEntity)
                .collect(java.util.stream.Collectors.toList());
    }
}