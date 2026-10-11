package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.QcPlanItem;
import com.metawebthree.mes.domain.repository.QcPlanItemRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.QcPlanItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.QcPlanItemMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class QcPlanItemRepositoryImpl implements QcPlanItemRepository {
    
    private final QcPlanItemMapper qcPlanItemMapper;
    
    public QcPlanItemRepositoryImpl(QcPlanItemMapper qcPlanItemMapper) {
        this.qcPlanItemMapper = qcPlanItemMapper;
    }
    
    @Override
    public QcPlanItem save(QcPlanItem entity) {
        QcPlanItemDO dto = toDO(entity);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            dto.setUpdatedAt(LocalDateTime.now());
            qcPlanItemMapper.insert(dto);
        } else {
            dto.setUpdatedAt(LocalDateTime.now());
            qcPlanItemMapper.updateById(dto);
        }
        return toDomain(dto);
    }
    
    @Override
    public Optional<QcPlanItem> findById(Long id) {
        QcPlanItemDO dto = qcPlanItemMapper.selectById(id);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public List<QcPlanItem> findByPlanId(Long planId) {
        LambdaQueryWrapper<QcPlanItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcPlanItemDO::getPlanId, planId);
        wrapper.orderByAsc(QcPlanItemDO::getSortOrder);
        return qcPlanItemMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcPlanItem> findByItemId(Long itemId) {
        LambdaQueryWrapper<QcPlanItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcPlanItemDO::getItemId, itemId);
        return qcPlanItemMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public void deleteById(Long id) {
        qcPlanItemMapper.deleteById(id);
    }
    
    @Override
    public void deleteByPlanId(Long planId) {
        LambdaQueryWrapper<QcPlanItemDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcPlanItemDO::getPlanId, planId);
        qcPlanItemMapper.delete(wrapper);
    }
    
    private QcPlanItem toDomain(QcPlanItemDO dto) {
        if (dto == null) return null;
        
        QcPlanItem entity = new QcPlanItem();
        entity.setId(dto.getId());
        entity.setPlanId(dto.getPlanId());
        entity.setItemId(dto.getItemId());
        entity.setItemSequence(dto.getItemSequence());
        entity.setIsMandatory(dto.getIsMandatory());
        entity.setDefaultValue(dto.getDefaultValue());
        entity.setInspectionMethod(dto.getInspectionMethod());
        entity.setSamplingRule(dto.getSamplingRule());
        entity.setSortOrder(dto.getSortOrder());
        return entity;
    }
    
    private QcPlanItemDO toDO(QcPlanItem entity) {
        if (entity == null) return null;
        
        QcPlanItemDO dto = new QcPlanItemDO();
        dto.setId(entity.getId());
        dto.setPlanId(entity.getPlanId());
        dto.setItemId(entity.getItemId());
        dto.setItemSequence(entity.getItemSequence());
        dto.setIsMandatory(entity.getIsMandatory());
        dto.setDefaultValue(entity.getDefaultValue());
        dto.setInspectionMethod(entity.getInspectionMethod());
        dto.setSamplingRule(entity.getSamplingRule());
        dto.setSortOrder(entity.getSortOrder());
        return dto;
    }
}