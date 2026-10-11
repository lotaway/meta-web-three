package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.QcInspectionPlan;
import com.metawebthree.mes.domain.repository.QcInspectionPlanRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.QcInspectionPlanDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.QcInspectionPlanMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class QcInspectionPlanRepositoryImpl implements QcInspectionPlanRepository {
    
    private final QcInspectionPlanMapper qcInspectionPlanMapper;
    
    public QcInspectionPlanRepositoryImpl(QcInspectionPlanMapper qcInspectionPlanMapper) {
        this.qcInspectionPlanMapper = qcInspectionPlanMapper;
    }
    
    @Override
    public QcInspectionPlan save(QcInspectionPlan entity) {
        QcInspectionPlanDO dto = toDO(entity);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            dto.setUpdatedAt(LocalDateTime.now());
            qcInspectionPlanMapper.insert(dto);
        } else {
            dto.setUpdatedAt(LocalDateTime.now());
            qcInspectionPlanMapper.updateById(dto);
        }
        return toDomain(dto);
    }
    
    @Override
    public Optional<QcInspectionPlan> findById(Long id) {
        QcInspectionPlanDO dto = qcInspectionPlanMapper.selectById(id);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<QcInspectionPlan> findByPlanCode(String planCode) {
        LambdaQueryWrapper<QcInspectionPlanDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionPlanDO::getPlanCode, planCode);
        QcInspectionPlanDO dto = qcInspectionPlanMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public List<QcInspectionPlan> findAll() {
        return qcInspectionPlanMapper.selectList(null).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcInspectionPlan> findByInspectionType(String inspectionType) {
        LambdaQueryWrapper<QcInspectionPlanDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionPlanDO::getInspectionType, inspectionType);
        return qcInspectionPlanMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<QcInspectionPlan> findByStatus(QcInspectionPlan.PlanStatus status) {
        LambdaQueryWrapper<QcInspectionPlanDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionPlanDO::getStatus, status.name());
        return qcInspectionPlanMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public void deleteById(Long id) {
        qcInspectionPlanMapper.deleteById(id);
    }
    
    @Override
    public boolean existsByPlanCode(String planCode) {
        LambdaQueryWrapper<QcInspectionPlanDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(QcInspectionPlanDO::getPlanCode, planCode);
        return qcInspectionPlanMapper.selectCount(wrapper) > 0;
    }
    
    private QcInspectionPlan toDomain(QcInspectionPlanDO dto) {
        if (dto == null) return null;
        
        QcInspectionPlan entity = new QcInspectionPlan();
        entity.setId(dto.getId());
        entity.setPlanCode(dto.getPlanCode());
        entity.setPlanName(dto.getPlanName());
        entity.setInspectionType(dto.getInspectionType());
        entity.setApplicableProductTypes(dto.getApplicableProductTypes());
        entity.setVersion(dto.getVersion());
        entity.setSamplingPlanCode(dto.getSamplingPlanCode());
        entity.setSamplingType(dto.getSamplingType());
        entity.setAql(dto.getAql());
        entity.setInspectionLevel(dto.getInspectionLevel());
        entity.setSampleSize(dto.getSampleSize());
        entity.setAcceptNumber(dto.getAcceptNumber());
        entity.setRejectNumber(dto.getRejectNumber());
        entity.setDispositionRule(dto.getDispositionRule());
        entity.setQualifiedFlow(dto.getQualifiedFlow());
        entity.setUnqualifiedFlow(dto.getUnqualifiedFlow());
        entity.setSpecialApprovalFlow(dto.getSpecialApprovalFlow());
        entity.setStatus(QcInspectionPlan.PlanStatus.valueOf(dto.getStatus()));
        entity.setEffectiveDate(dto.getEffectiveDate());
        entity.setExpirationDate(dto.getExpirationDate());
        entity.setSortOrder(dto.getSortOrder());
        entity.setRemark(dto.getRemark());
        return entity;
    }
    
    private QcInspectionPlanDO toDO(QcInspectionPlan entity) {
        if (entity == null) return null;
        
        QcInspectionPlanDO dto = new QcInspectionPlanDO();
        dto.setId(entity.getId());
        dto.setPlanCode(entity.getPlanCode());
        dto.setPlanName(entity.getPlanName());
        dto.setInspectionType(entity.getInspectionType());
        dto.setApplicableProductTypes(entity.getApplicableProductTypes());
        dto.setVersion(entity.getVersion());
        dto.setSamplingPlanCode(entity.getSamplingPlanCode());
        dto.setSamplingType(entity.getSamplingType());
        dto.setAql(entity.getAql());
        dto.setInspectionLevel(entity.getInspectionLevel());
        dto.setSampleSize(entity.getSampleSize());
        dto.setAcceptNumber(entity.getAcceptNumber());
        dto.setRejectNumber(entity.getRejectNumber());
        dto.setDispositionRule(entity.getDispositionRule());
        dto.setQualifiedFlow(entity.getQualifiedFlow());
        dto.setUnqualifiedFlow(entity.getUnqualifiedFlow());
        dto.setSpecialApprovalFlow(entity.getSpecialApprovalFlow());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setEffectiveDate(entity.getEffectiveDate());
        dto.setExpirationDate(entity.getExpirationDate());
        dto.setSortOrder(entity.getSortOrder());
        dto.setRemark(entity.getRemark());
        return dto;
    }
}