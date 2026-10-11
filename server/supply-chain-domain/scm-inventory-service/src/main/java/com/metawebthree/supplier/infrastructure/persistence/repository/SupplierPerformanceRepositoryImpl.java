package com.metawebthree.supplier.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.supplier.domain.entity.SupplierPerformance;
import com.metawebthree.supplier.domain.repository.SupplierPerformanceRepository;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierPerformanceDO;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierPerformanceMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 供应商绩效评估仓储实现
 */
@Repository
public class SupplierPerformanceRepositoryImpl implements SupplierPerformanceRepository {
    
    private final SupplierPerformanceMapper supplierPerformanceMapper;
    
    public SupplierPerformanceRepositoryImpl(SupplierPerformanceMapper supplierPerformanceMapper) {
        this.supplierPerformanceMapper = supplierPerformanceMapper;
    }
    
    @Override
    public SupplierPerformance save(SupplierPerformance performance) {
        SupplierPerformanceDO dataObject = toDO(performance);
        dataObject.setCreatedAt(java.time.LocalDateTime.now());
        dataObject.setUpdatedAt(java.time.LocalDateTime.now());
        supplierPerformanceMapper.insert(dataObject);
        return toEntity(dataObject);
    }
    
    @Override
    public SupplierPerformance findById(Long id) {
        SupplierPerformanceDO dataObject = supplierPerformanceMapper.selectById(id);
        return dataObject != null ? toEntity(dataObject) : null;
    }
    
    @Override
    public List<SupplierPerformance> findBySupplierId(Long supplierId) {
        LambdaQueryWrapper<SupplierPerformanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SupplierPerformanceDO::getSupplierId, supplierId)
               .orderByDesc(SupplierPerformanceDO::getAssessmentDate);
        return supplierPerformanceMapper.selectList(wrapper).stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
    
    @Override
    public SupplierPerformance findBySupplierIdAndPeriod(Long supplierId, LocalDate periodStart, LocalDate periodEnd) {
        LambdaQueryWrapper<SupplierPerformanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SupplierPerformanceDO::getSupplierId, supplierId)
               .eq(SupplierPerformanceDO::getPeriodStart, periodStart)
               .eq(SupplierPerformanceDO::getPeriodEnd, periodEnd);
        SupplierPerformanceDO dataObject = supplierPerformanceMapper.selectOne(wrapper);
        return dataObject != null ? toEntity(dataObject) : null;
    }
    
    @Override
    public List<SupplierPerformance> findAll() {
        return supplierPerformanceMapper.selectList(null).stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<SupplierPerformance> findByAssessmentLevel(String assessmentLevel) {
        LambdaQueryWrapper<SupplierPerformanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SupplierPerformanceDO::getAssessmentLevel, assessmentLevel)
               .orderByDesc(SupplierPerformanceDO::getOverallScore);
        return supplierPerformanceMapper.selectList(wrapper).stream()
                .map(this::toEntity)
                .collect(Collectors.toList());
    }
    
    @Override
    public void deleteById(Long id) {
        supplierPerformanceMapper.deleteById(id);
    }
    
    @Override
    public SupplierPerformance update(SupplierPerformance performance) {
        SupplierPerformanceDO dataObject = toDO(performance);
        dataObject.setUpdatedAt(java.time.LocalDateTime.now());
        supplierPerformanceMapper.updateById(dataObject);
        return toEntity(dataObject);
    }
    
    @Override
    public List<SupplierPerformance> batchSave(List<SupplierPerformance> performances) {
        return performances.stream()
                .map(this::save)
                .collect(Collectors.toList());
    }
    
    private SupplierPerformanceDO toDO(SupplierPerformance entity) {
        if (entity == null) {
            return null;
        }
        SupplierPerformanceDO dataObject = new SupplierPerformanceDO();
        BeanUtils.copyProperties(entity, dataObject);
        return dataObject;
    }
    
    private SupplierPerformance toEntity(SupplierPerformanceDO dataObject) {
        if (dataObject == null) {
            return null;
        }
        SupplierPerformance entity = new SupplierPerformance();
        BeanUtils.copyProperties(dataObject, entity);
        return entity;
    }
}