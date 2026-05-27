package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.WorkOrderCodeRule;

import java.util.List;
import java.util.Optional;

public interface WorkOrderCodeRuleRepository {
    
    Optional<WorkOrderCodeRule> findById(Long id);
    
    WorkOrderCodeRule save(WorkOrderCodeRule binding);
    
    void deleteById(Long id);
    
    Optional<WorkOrderCodeRule> findActiveByWorkshopAndType(String workshopId, String workOrderType);
    
    Optional<WorkOrderCodeRule> findActiveByType(String workOrderType);
    
    Optional<WorkOrderCodeRule> findActiveByWorkshop(String workshopId);
    
    List<WorkOrderCodeRule> findAllActive();
    
    Optional<WorkOrderCodeRule> findActiveByRuleCode(String ruleCode);
}