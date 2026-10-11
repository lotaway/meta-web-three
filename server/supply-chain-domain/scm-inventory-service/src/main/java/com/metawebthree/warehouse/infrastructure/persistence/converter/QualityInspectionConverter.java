package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.QualityInspection;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityInspectionDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class QualityInspectionConverter {
    
    public QualityInspection toEntity(QualityInspectionDO dto) {
        if (dto == null) {
            return null;
        }
        QualityInspection entity = new QualityInspection();
        entity.setId(dto.getId());
        entity.setInspectionNo(dto.getInspectionNo());
        entity.setOrderId(dto.getOrderId());
        entity.setOrderNo(dto.getOrderNo());
        entity.setInboundType(dto.getInboundType());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setSupplierCode(dto.getSupplierCode());
        entity.setSupplierName(dto.getSupplierName());
        entity.setInspectionType(dto.getInspectionType());
        entity.setInspectionStatus(dto.getInspectionStatus());
        entity.setTotalQuantity(dto.getTotalQuantity());
        entity.setInspectedQuantity(dto.getInspectedQuantity());
        entity.setQualifiedQuantity(dto.getQualifiedQuantity());
        entity.setUnqualifiedQuantity(dto.getUnqualifiedQuantity());
        entity.setConcessionQuantity(dto.getConcessionQuantity());
        entity.setDefectRate(dto.getDefectRate());
        entity.setInspector(dto.getInspector());
        entity.setInspectionTime(dto.getInspectionTime());
        entity.setResultRemark(dto.getResultRemark());
        entity.setIsAutoInspection(dto.getIsAutoInspection() != null && dto.getIsAutoInspection() == 1);
        entity.setSourceSystem(dto.getSourceSystem());
        entity.setCreator(dto.getCreator());
        entity.setCreateTime(dto.getCreateTime());
        entity.setUpdater(dto.getUpdater());
        entity.setUpdateTime(dto.getUpdateTime());
        entity.setDeleted(dto.getDeleted() != null && dto.getDeleted() == 1);
        return entity;
    }
    
    public QualityInspectionDO toDO(QualityInspection entity) {
        if (entity == null) {
            return null;
        }
        QualityInspectionDO dto = new QualityInspectionDO();
        dto.setId(entity.getId());
        dto.setInspectionNo(entity.getInspectionNo());
        dto.setOrderId(entity.getOrderId());
        dto.setOrderNo(entity.getOrderNo());
        dto.setInboundType(entity.getInboundType());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setSupplierCode(entity.getSupplierCode());
        dto.setSupplierName(entity.getSupplierName());
        dto.setInspectionType(entity.getInspectionType());
        dto.setInspectionStatus(entity.getInspectionStatus());
        dto.setTotalQuantity(entity.getTotalQuantity());
        dto.setInspectedQuantity(entity.getInspectedQuantity());
        dto.setQualifiedQuantity(entity.getQualifiedQuantity());
        dto.setUnqualifiedQuantity(entity.getUnqualifiedQuantity());
        dto.setConcessionQuantity(entity.getConcessionQuantity());
        dto.setDefectRate(entity.getDefectRate());
        dto.setInspector(entity.getInspector());
        dto.setInspectionTime(entity.getInspectionTime());
        dto.setResultRemark(entity.getResultRemark());
        dto.setIsAutoInspection(entity.getIsAutoInspection() != null && entity.getIsAutoInspection() ? 1 : 0);
        dto.setSourceSystem(entity.getSourceSystem());
        dto.setCreator(entity.getCreator());
        dto.setCreateTime(entity.getCreateTime());
        dto.setUpdater(entity.getUpdater());
        dto.setUpdateTime(entity.getUpdateTime());
        dto.setDeleted(entity.getDeleted() != null && entity.getDeleted() ? 1 : 0);
        return dto;
    }
    
    public List<QualityInspection> toEntityList(List<QualityInspectionDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        return dtoList.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
}