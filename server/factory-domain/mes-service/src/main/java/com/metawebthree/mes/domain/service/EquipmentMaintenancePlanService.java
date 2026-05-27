package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.EquipmentMaintenancePlan;
import com.metawebthree.mes.domain.repository.EquipmentMaintenancePlanRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class EquipmentMaintenancePlanService {
    
    private final EquipmentMaintenancePlanRepository planRepository;
    
    public EquipmentMaintenancePlanService(EquipmentMaintenancePlanRepository planRepository) {
        this.planRepository = planRepository;
    }
    
    public EquipmentMaintenancePlan createTimeBasedPlan(String planCode, String planName, 
            String equipmentTypeCode, Integer cycleDays, Integer advanceAlertDays) {
        EquipmentMaintenancePlan plan = new EquipmentMaintenancePlan();
        plan.create(planCode, planName, equipmentTypeCode, EquipmentMaintenancePlan.MaintenanceCycleType.TIME_BASED);
        plan.setTimeBasedCycle(cycleDays, advanceAlertDays);
        return planRepository.save(plan);
    }
    
    public EquipmentMaintenancePlan createRunningHoursPlan(String planCode, String planName,
            String equipmentTypeCode, Integer cycleRunningHours, Integer advanceAlertDays) {
        EquipmentMaintenancePlan plan = new EquipmentMaintenancePlan();
        plan.create(planCode, planName, equipmentTypeCode, EquipmentMaintenancePlan.MaintenanceCycleType.RUNNING_HOURS);
        plan.setRunningHoursCycle(cycleRunningHours, advanceAlertDays);
        return planRepository.save(plan);
    }
    
    public Optional<EquipmentMaintenancePlan> getPlanById(Long id) {
        return planRepository.findById(id);
    }
    
    public Optional<EquipmentMaintenancePlan> getPlanByCode(String planCode) {
        return planRepository.findByPlanCode(planCode);
    }
    
    public List<EquipmentMaintenancePlan> getAllPlans() {
        return planRepository.findAll();
    }
    
    public List<EquipmentMaintenancePlan> getPlansByEquipmentType(String equipmentTypeCode) {
        return planRepository.findByEquipmentTypeCode(equipmentTypeCode);
    }
    
    public List<EquipmentMaintenancePlan> getActivePlans() {
        return planRepository.findByIsActive(true);
    }
    
    public EquipmentMaintenancePlan addItem(Long planId, EquipmentMaintenancePlan.MaintenanceItem item) {
        Optional<EquipmentMaintenancePlan> optPlan = planRepository.findById(planId);
        if (optPlan.isEmpty()) {
            return null;
        }
        EquipmentMaintenancePlan plan = optPlan.get();
        plan.addItem(item);
        planRepository.update(plan);
        return planRepository.findById(planId).orElse(plan);
    }
    
    public EquipmentMaintenancePlan updatePlan(EquipmentMaintenancePlan plan) {
        planRepository.update(plan);
        return plan;
    }
    
    public void deletePlan(Long id) {
        planRepository.deleteById(id);
    }
    
    public void activatePlan(Long id) {
        planRepository.findById(id).ifPresent(plan -> {
            plan.activate();
            planRepository.update(plan);
        });
    }
    
    public void deactivatePlan(Long id) {
        planRepository.findById(id).ifPresent(plan -> {
            plan.deactivate();
            planRepository.update(plan);
        });
    }
}