package com.metawebthree.supplier.domain.repository;

import com.metawebthree.supplier.domain.entity.SupplierPerformance;
import java.time.LocalDate;
import java.util.List;

/**
 * 供应商绩效评估仓储接口
 */
public interface SupplierPerformanceRepository {
    
    /**
     * 保存绩效评估记录
     */
    SupplierPerformance save(SupplierPerformance performance);
    
    /**
     * 根据ID查询绩效评估记录
     */
    SupplierPerformance findById(Long id);
    
    /**
     * 根据供应商ID查询绩效评估记录列表
     */
    List<SupplierPerformance> findBySupplierId(Long supplierId);
    
    /**
     * 根据供应商ID和评估周期查询绩效评估记录
     */
    SupplierPerformance findBySupplierIdAndPeriod(Long supplierId, LocalDate periodStart, LocalDate periodEnd);
    
    /**
     * 查询所有绩效评估记录
     */
    List<SupplierPerformance> findAll();
    
    /**
     * 根据评估等级查询绩效评估记录
     */
    List<SupplierPerformance> findByAssessmentLevel(String assessmentLevel);
    
    /**
     * 删除绩效评估记录
     */
    void deleteById(Long id);
    
    /**
     * 更新绩效评估记录
     */
    SupplierPerformance update(SupplierPerformance performance);
    
    /**
     * 批量保存绩效评估记录
     */
    List<SupplierPerformance> batchSave(List<SupplierPerformance> performances);
}