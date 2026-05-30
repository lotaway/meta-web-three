package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.QualityInspectionItem;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityInspectionItemDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class QualityInspectionItemConverter {
    
    public QualityInspectionItem toEntity(QualityInspectionItemDO dto) {
        if (dto == null) {
            return null;
        }
        QualityInspectionItem entity = new QualityInspectionItem();
        entity.setId(dto.getId());
        entity.setInspectionId(dto.getInspectionId());
        entity.setSkuCode(dto.getSkuCode());
        entity.setProductName(dto.getProductName());
        entity.setBatchNo(dto.getBatchNo());
        entity.setLocationCode(dto.getLocationCode());
        entity.setPlanQuantity(dto.getPlanQuantity());
        entity.setActualQuantity(dto.getActualQuantity());
        entity.setInspectedQuantity(dto.getInspectedQuantity());
        entity.setQualifiedQuantity(dto.getQualifiedQuantity());
        entity.setUnqualifiedQuantity(dto.getUnqualifiedQuantity());
        entity.setConcessionQuantity(dto.getConcessionQuantity());
        entity.setSampleQuantity(dto.getSampleQuantity());
        entity.setDefectItems(dto.getDefectItems());
        entity.setCheckResult(dto.getCheckResult());
        entity.setRemark(dto.getRemark());
        entity.setCreator(dto.getCreator());
        entity.setCreateTime(dto.getCreateTime());
        entity.setUpdater(dto.getUpdater());
        entity.setUpdateTime(dto.getUpdateTime());
        entity.setDeleted(dto.getDeleted() != null && dto.getDeleted() == 1);
        return entity;
    }
    
    public QualityInspectionItemDO toDO(QualityInspectionItem entity) {
        if (entity == null) {
            return null;
        }
        QualityInspectionItemDO dto = new QualityInspectionItemDO();
        dto.setId(entity.getId());
        dto.setInspectionId(entity.getInspectionId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setProductName(entity.getProductName());
        dto.setBatchNo(entity.getBatchNo());
        dto.setLocationCode(entity.getLocationCode());
        dto.setPlanQuantity(entity.getPlanQuantity());
        dto.setActualQuantity(entity.getActualQuantity());
        dto.setInspectedQuantity(entity.getInspectedQuantity());
        dto.setQualifiedQuantity(entity.getQualifiedQuantity());
        dto.setUnqualifiedQuantity(entity.getUnqualifiedQuantity());
        dto.setConcessionQuantity(entity.getConcessionQuantity());
        dto.setSampleQuantity(entity.getSampleQuantity());
        dto.setDefectItems(entity.getDefectItems());
        dto.setCheckResult(entity.getCheckResult());
        dto.setRemark(entity.getRemark());
        dto.setCreator(entity.getCreator());
        dto.setCreateTime(entity.getCreateTime());
        dto.setUpdater(entity.getUpdater());
        dto.setUpdateTime(entity.getUpdateTime());
        dto.setDeleted(entity.getDeleted() != null && entity.getDeleted() ? 1 : 0);
        return dto;
    }
    
    public List<QualityInspectionItem> toEntityList(List<QualityInspectionItemDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        return dtoList.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
}