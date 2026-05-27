package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.QcInspectionItem;
import com.metawebthree.mes.domain.repository.QcInspectionItemRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.QcInspectionItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.QcInspectionItemMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class QcInspectionItemRepositoryImpl implements QcInspectionItemRepository {
    
    private final QcInspectionItemMapper qcInspectionItemMapper;
    
    public QcInspectionItemRepositoryImpl(QcInspectionItemMapper qcInspectionItemMapper) {
        this.qcInspectionItemMapper = qcInspectionItemMapper;
    }
    
    @Override
    public QcInspectionItem save(QcInspectionItem entity) {
        QcInspectionItemDO dto = toDO(entity);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            dto.setUpdatedAt(LocalDateTime.now());
            qcInspectionItemMapper.insert(dto);
        } else {
            dto.setUpdatedAt(LocalDateTime.now());
            qcInspectionItemMapper.updateById(dto);
        }
        return toDomain(dto);
    }
    
    @Override
    public Optional<QcInspectionItem> findById(Long id) {
        QcInspectionItemDO dto = qcInspectionItemMapper.selectById(id);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<QcInspectionItem> findByItemCode(String itemCode) {
        LambdaQueryWrapper<QcInspectionItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionItemDO::getItemCode, itemCode);
        QcInspectionItemDO dto = qcInspectionItemMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public List<QcInspectionItem> findAll() {
        return qcInspectionItemMapper.selectList(null).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcInspectionItem> findByItemCategory(String itemCategory) {
        LambdaQueryWrapper<QcInspectionItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionItemDO::getItemCategory, itemCategory);
        return qcInspectionItemMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcInspectionItem> findByStatus(QcInspectionItem.ItemStatus status) {
        LambdaQueryWrapper<QcInspectionItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionItemDO::getStatus, status.name());
        return qcInspectionItemMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public void deleteById(Long id) {
        qcInspectionItemMapper.deleteById(id);
    }
    
    @Override
    public boolean existsByItemCode(String itemCode) {
        LambdaQueryWrapper<QcInspectionItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionItemDO::getItemCode, itemCode);
        return qcInspectionItemMapper.selectCount(wrapper) > 0;
    }
    
    private QcInspectionItem toDomain(QcInspectionItemDO dto) {
        if (dto == null) return null;
        
        QcInspectionItem entity = new QcInspectionItem();
        entity.setId(dto.getId());
        entity.setItemCode(dto.getItemCode());
        entity.setItemName(dto.getItemName());
        entity.setItemCategory(dto.getItemCategory());
        entity.setDataType(dto.getDataType());
        entity.setUnit(dto.getUnit());
        entity.setStandardValue(dto.getStandardValue());
        entity.setUpperLimit(dto.getUpperLimit());
        entity.setLowerLimit(dto.getLowerLimit());
        entity.setInspectionMethod(dto.getInspectionMethod());
        entity.setInspectionTool(dto.getInspectionTool());
        entity.setSeverity(dto.getSeverity());
        entity.setIsMandatory(dto.getIsMandatory());
        entity.setStatus(Enum.valueOf(QcInspectionItem.ItemStatus.class, dto.getStatus()));
        entity.setSortOrder(dto.getSortOrder());
        entity.setRemark(dto.getRemark());
        return entity;
    }
    
    private QcInspectionItemDO toDO(QcInspectionItem entity) {
        if (entity == null) return null;
        
        QcInspectionItemDO dto = new QcInspectionItemDO();
        dto.setId(entity.getId());
        dto.setItemCode(entity.getItemCode());
        dto.setItemName(entity.getItemName());
        dto.setItemCategory(entity.getItemCategory());
        dto.setDataType(entity.getDataType());
        dto.setUnit(entity.getUnit());
        dto.setStandardValue(entity.getStandardValue());
        dto.setUpperLimit(entity.getUpperLimit());
        dto.setLowerLimit(entity.getLowerLimit());
        dto.setInspectionMethod(entity.getInspectionMethod());
        dto.setInspectionTool(entity.getInspectionTool());
        dto.setSeverity(entity.getSeverity());
        dto.setIsMandatory(entity.getIsMandatory());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setSortOrder(entity.getSortOrder());
        dto.setRemark(entity.getRemark());
        return dto;
    }
}