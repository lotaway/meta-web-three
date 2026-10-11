package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.QcInspectionType;
import com.metawebthree.mes.domain.repository.QcInspectionTypeRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.QcInspectionTypeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.QcInspectionTypeMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class QcInspectionTypeRepositoryImpl implements QcInspectionTypeRepository {
    
    private final QcInspectionTypeMapper qcInspectionTypeMapper;
    
    public QcInspectionTypeRepositoryImpl(QcInspectionTypeMapper qcInspectionTypeMapper) {
        this.qcInspectionTypeMapper = qcInspectionTypeMapper;
    }
    
    @Override
    public QcInspectionType save(QcInspectionType entity) {
        QcInspectionTypeDO dto = toDO(entity);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            dto.setUpdatedAt(LocalDateTime.now());
            qcInspectionTypeMapper.insert(dto);
        } else {
            dto.setUpdatedAt(LocalDateTime.now());
            qcInspectionTypeMapper.updateById(dto);
        }
        return toDomain(dto);
    }
    
    @Override
    public Optional<QcInspectionType> findById(Long id) {
        QcInspectionTypeDO dto = qcInspectionTypeMapper.selectById(id);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<QcInspectionType> findByTypeCode(String typeCode) {
        LambdaQueryWrapper<QcInspectionTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionTypeDO::getTypeCode, typeCode);
        QcInspectionTypeDO dto = qcInspectionTypeMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public List<QcInspectionType> findAll() {
        return qcInspectionTypeMapper.selectList(null).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcInspectionType> findByCategory(QcInspectionType.InspectionCategory category) {
        LambdaQueryWrapper<QcInspectionTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionTypeDO::getCategory, category.name());
        return qcInspectionTypeMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcInspectionType> findByStatus(QcInspectionType.InspectionStatus status) {
        LambdaQueryWrapper<QcInspectionTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionTypeDO::getStatus, status.name());
        return qcInspectionTypeMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public void deleteById(Long id) {
        qcInspectionTypeMapper.deleteById(id);
    }
    
    @Override
    public boolean existsByTypeCode(String typeCode) {
        LambdaQueryWrapper<QcInspectionTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionTypeDO::getTypeCode, typeCode);
        return qcInspectionTypeMapper.selectCount(wrapper) > 0;
    }
    
    private QcInspectionType toDomain(QcInspectionTypeDO dto) {
        if (dto == null) return null;
        
        QcInspectionType entity = new QcInspectionType();
        entity.setId(dto.getId());
        entity.setTypeCode(dto.getTypeCode());
        entity.setTypeName(dto.getTypeName());
        entity.setCategory(QcInspectionType.InspectionCategory.valueOf(dto.getCategory()));
        entity.setDescription(dto.getDescription());
        entity.setApplicableProducts(dto.getApplicableProducts());
        entity.setDefaultSamplingPlan(dto.getDefaultSamplingPlan());
        entity.setDefaultAql(dto.getDefaultAql());
        entity.setDefaultTimeoutHours(dto.getDefaultTimeoutHours());
        entity.setRequireCertificate(dto.getRequireCertificate());
        entity.setRequireTestReport(dto.getRequireTestReport());
        entity.setStatus(QcInspectionType.InspectionStatus.valueOf(dto.getStatus()));
        entity.setSortOrder(dto.getSortOrder());
        return entity;
    }
    
    private QcInspectionTypeDO toDO(QcInspectionType entity) {
        if (entity == null) return null;
        
        QcInspectionTypeDO dto = new QcInspectionTypeDO();
        dto.setId(entity.getId());
        dto.setTypeCode(entity.getTypeCode());
        dto.setTypeName(entity.getTypeName());
        dto.setCategory(entity.getCategory() != null ? entity.getCategory().name() : null);
        dto.setDescription(entity.getDescription());
        dto.setApplicableProducts(entity.getApplicableProducts());
        dto.setDefaultSamplingPlan(entity.getDefaultSamplingPlan());
        dto.setDefaultAql(entity.getDefaultAql());
        dto.setDefaultTimeoutHours(entity.getDefaultTimeoutHours());
        dto.setRequireCertificate(entity.getRequireCertificate());
        dto.setRequireTestReport(entity.getRequireTestReport());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setSortOrder(entity.getSortOrder());
        return dto;
    }
}