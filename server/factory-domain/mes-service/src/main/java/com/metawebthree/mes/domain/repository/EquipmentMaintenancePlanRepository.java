package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EquipmentMaintenancePlan;

import java.util.List;
import java.util.Optional;

public interface EquipmentMaintenancePlanRepository {
    
    EquipmentMaintenancePlan save(EquipmentMaintenancePlan plan);
    
    Optional<EquipmentMaintenancePlan> findById(Long id);
    
    Optional<EquipmentMaintenancePlan> findByPlanCode(String planCode);
    
    List<EquipmentMaintenancePlan> findAll();
    
    List<EquipmentMaintenancePlan> findByEquipmentTypeCode(String equipmentTypeCode);
    
    List<EquipmentMaintenancePlan> findByIsActive(Boolean isActive);
    
    void update(EquipmentMaintenancePlan plan);
    
    void deleteById(Long id);
}