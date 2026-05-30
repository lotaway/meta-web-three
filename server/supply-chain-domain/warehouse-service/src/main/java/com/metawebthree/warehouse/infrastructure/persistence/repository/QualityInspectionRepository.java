package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.metawebthree.warehouse.domain.entity.QualityInspection;
import com.metawebthree.warehouse.domain.entity.QualityInspectionItem;
import com.metawebthree.warehouse.infrastructure.persistence.converter.QualityInspectionConverter;
import com.metawebthree.warehouse.infrastructure.persistence.converter.QualityInspectionItemConverter;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityInspectionDO;
import com.metawebthree.warehouse.infrastructure.persistence.dataobject.QualityInspectionItemDO;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.QualityInspectionMapper;
import com.metawebthree.warehouse.infrastructure.persistence.mapper.QualityInspectionItemMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public class QualityInspectionRepository {
    
    @Autowired
    private QualityInspectionMapper mapper;
    
    @Autowired
    private QualityInspectionItemMapper itemMapper;
    
    @Autowired
    private QualityInspectionConverter converter;
    
    @Autowired
    private QualityInspectionItemConverter itemConverter;
    
    public QualityInspection save(QualityInspection entity) {
        QualityInspectionDO dto = converter.toDO(entity);
        dto.setCreateTime(LocalDateTime.now());
        dto.setUpdateTime(LocalDateTime.now());
        mapper.insert(dto);
        entity.setId(dto.getId());
        return entity;
    }
    
    public QualityInspection update(QualityInspection entity) {
        QualityInspectionDO dto = converter.toDO(entity);
        dto.setUpdateTime(LocalDateTime.now());
        mapper.updateById(dto);
        return entity;
    }
    
    public QualityInspection findById(Long id) {
        QualityInspectionDO dto = mapper.selectById(id);
        return converter.toEntity(dto);
    }
    
    public QualityInspection findByInspectionNo(String inspectionNo) {
        QualityInspectionDO dto = mapper.selectByInspectionNo(inspectionNo);
        return converter.toEntity(dto);
    }
    
    public List<QualityInspection> findByOrderId(Long orderId) {
        List<QualityInspectionDO> dtoList = mapper.selectByOrderId(orderId);
        return converter.toEntityList(dtoList);
    }
    
    public List<QualityInspection> findAll() {
        LambdaQueryWrapper<QualityInspectionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QualityInspectionDO::getDeleted, 0)
               .orderByDesc(QualityInspectionDO::getId);
        List<QualityInspectionDO> dtoList = mapper.selectList(wrapper);
        return converter.toEntityList(dtoList);
    }
    
    // Item operations
    public QualityInspectionItem saveItem(QualityInspectionItem item) {
        QualityInspectionItemDO dto = itemConverter.toDO(item);
        dto.setCreateTime(LocalDateTime.now());
        dto.setUpdateTime(LocalDateTime.now());
        itemMapper.insert(dto);
        item.setId(dto.getId());
        return item;
    }
    
    public List<QualityInspectionItem> findItemsByInspectionId(Long inspectionId) {
        List<QualityInspectionItemDO> dtoList = itemMapper.selectByInspectionId(inspectionId);
        return itemConverter.toEntityList(dtoList);
    }
    
    private List<QualityInspection> toEntityList(List<QualityInspectionDO> dtoList) {
        if (dtoList == null) return null;
        return dtoList.stream().map(converter::toEntity).collect(java.util.stream.Collectors.toList());
    }
}