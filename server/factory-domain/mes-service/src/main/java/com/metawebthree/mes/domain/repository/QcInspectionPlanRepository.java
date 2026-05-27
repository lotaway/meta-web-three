package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.QcInspectionPlan;
import java.util.List;
import java.util.Optional;

public interface QcInspectionPlanRepository {
    
    QcInspectionPlan save(QcInspectionPlan entity);
    
    Optional<QcInspectionPlan> findById(Long id);
    
    Optional<QcInspectionPlan> findByPlanCode(String planCode);
    
    List<QcInspectionPlan> findAll();
    
    List<QcInspectionPlan> findByInspectionType(String inspectionType);
    
    List<QcInspectionPlan> findByStatus(QcInspectionPlan.PlanStatus status);
    
    void deleteById(Long id);
    
    boolean existsByPlanCode(String planCode);
}