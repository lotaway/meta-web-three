package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.DefectProcessing;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.DefectProcessingDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class DefectProcessingConverter {
    
    public DefectProcessing toEntity(DefectProcessingDO dto) {
        if (dto == null) {
            return null;
        }
        DefectProcessing entity = new DefectProcessing();
        entity.setId(dto.getId());
        entity.setDefectId(dto.getDefectId());
        entity.setProcessingNo(dto.getProcessingNo());
        entity.setProcessingType(dto.getProcessingType());
        entity.setProcessingStatus(dto.getProcessingStatus());
        entity.setProcessingQuantity(dto.getProcessingQuantity());
        entity.setProcessingPrice(dto.getProcessingPrice());
        entity.setProcessingReason(dto.getProcessingReason());
        entity.setProcessingRemark(dto.getProcessingRemark());
        entity.setProcessor(dto.getProcessor());
        entity.setProcessingTime(dto.getProcessingTime());
        entity.setRelatedDocumentNo(dto.getRelatedDocumentNo());
        entity.setRelatedDocumentType(dto.getRelatedDocumentType());
        entity.setApprover(dto.getApprover());
        entity.setApproveTime(dto.getApproveTime());
        entity.setApproveRemark(dto.getApproveRemark());
        entity.setCreator(dto.getCreator());
        entity.setCreateTime(dto.getCreateTime());
        entity.setUpdater(dto.getUpdater());
        entity.setUpdateTime(dto.getUpdateTime());
        entity.setDeleted(dto.getDeleted() != null && dto.getDeleted() == 1);
        return entity;
    }
    
    public DefectProcessingDO toDO(DefectProcessing entity) {
        if (entity == null) {
            return null;
        }
        DefectProcessingDO dto = new DefectProcessingDO();
        dto.setId(entity.getId());
        dto.setDefectId(entity.getDefectId());
        dto.setProcessingNo(entity.getProcessingNo());
        dto.setProcessingType(entity.getProcessingType());
        dto.setProcessingStatus(entity.getProcessingStatus());
        dto.setProcessingQuantity(entity.getProcessingQuantity());
        dto.setProcessingPrice(entity.getProcessingPrice());
        dto.setProcessingReason(entity.getProcessingReason());
        dto.setProcessingRemark(entity.getProcessingRemark());
        dto.setProcessor(entity.getProcessor());
        dto.setProcessingTime(entity.getProcessingTime());
        dto.setRelatedDocumentNo(entity.getRelatedDocumentNo());
        dto.setRelatedDocumentType(entity.getRelatedDocumentType());
        dto.setApprover(entity.getApprover());
        dto.setApproveTime(entity.getApproveTime());
        dto.setApproveRemark(entity.getApproveRemark());
        dto.setCreator(entity.getCreator());
        dto.setCreateTime(entity.getCreateTime());
        dto.setUpdater(entity.getUpdater());
        dto.setUpdateTime(entity.getUpdateTime());
        dto.setDeleted(entity.getDeleted() != null && entity.getDeleted() ? 1 : 0);
        return dto;
    }
    
    public List<DefectProcessing> toEntityList(List<DefectProcessingDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        return dtoList.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
}