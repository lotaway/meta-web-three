package com.metawebthree.warehouse.infrastructure.persistence.converter;

import com.metawebthree.warehouse.domain.entity.DefectRecord;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.DefectRecordDO;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

@Component
public class DefectRecordConverter {
    
    public DefectRecord toEntity(DefectRecordDO dto) {
        if (dto == null) {
            return null;
        }
        DefectRecord entity = new DefectRecord();
        entity.setId(dto.getId());
        entity.setInspectionId(dto.getInspectionId());
        entity.setInspectionItemId(dto.getInspectionItemId());
        entity.setSkuCode(dto.getSkuCode());
        entity.setProductName(dto.getProductName());
        entity.setBatchNo(dto.getBatchNo());
        entity.setDefectType(dto.getDefectType());
        entity.setDefectName(dto.getDefectName());
        entity.setDefectDescription(dto.getDefectDescription());
        entity.setDefectQuantity(dto.getDefectQuantity());
        entity.setDefectLevel(dto.getDefectLevel());
        entity.setPhotoUrls(dto.getPhotoUrls());
        entity.setLocationCode(dto.getLocationCode());
        entity.setCreator(dto.getCreator());
        entity.setCreateTime(dto.getCreateTime());
        entity.setUpdater(dto.getUpdater());
        entity.setUpdateTime(dto.getUpdateTime());
        entity.setDeleted(dto.getDeleted() != null && dto.getDeleted() == 1);
        return entity;
    }
    
    public DefectRecordDO toDO(DefectRecord entity) {
        if (entity == null) {
            return null;
        }
        DefectRecordDO dto = new DefectRecordDO();
        dto.setId(entity.getId());
        dto.setInspectionId(entity.getInspectionId());
        dto.setInspectionItemId(entity.getInspectionItemId());
        dto.setSkuCode(entity.getSkuCode());
        dto.setProductName(entity.getProductName());
        dto.setBatchNo(entity.getBatchNo());
        dto.setDefectType(entity.getDefectType());
        dto.setDefectName(entity.getDefectName());
        dto.setDefectDescription(entity.getDefectDescription());
        dto.setDefectQuantity(entity.getDefectQuantity());
        dto.setDefectLevel(entity.getDefectLevel());
        dto.setPhotoUrls(entity.getPhotoUrls());
        dto.setLocationCode(entity.getLocationCode());
        dto.setCreator(entity.getCreator());
        dto.setCreateTime(entity.getCreateTime());
        dto.setUpdater(entity.getUpdater());
        dto.setUpdateTime(entity.getUpdateTime());
        dto.setDeleted(entity.getDeleted() != null && entity.getDeleted() ? 1 : 0);
        return dto;
    }
    
    public List<DefectRecord> toEntityList(List<DefectRecordDO> dtoList) {
        if (dtoList == null) {
            return null;
        }
        return dtoList.stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
}