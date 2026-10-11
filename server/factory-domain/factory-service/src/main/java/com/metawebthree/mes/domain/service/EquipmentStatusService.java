package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.entity.EquipmentStatusConfig;
import com.metawebthree.mes.domain.entity.EquipmentStatusTransition;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import com.metawebthree.mes.domain.repository.EquipmentStatusConfigRepository;
import com.metawebthree.mes.domain.repository.EquipmentStatusTransitionRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class EquipmentStatusService {
    
    private final EquipmentRepository equipmentRepository;
    private final EquipmentStatusConfigRepository statusConfigRepository;
    private final EquipmentStatusTransitionRepository statusTransitionRepository;
    
    public EquipmentStatusService(
            EquipmentRepository equipmentRepository,
            EquipmentStatusConfigRepository statusConfigRepository,
            EquipmentStatusTransitionRepository statusTransitionRepository) {
        this.equipmentRepository = equipmentRepository;
        this.statusConfigRepository = statusConfigRepository;
        this.statusTransitionRepository = statusTransitionRepository;
    }
    
    public boolean validateTransition(Long equipmentId, String targetStatus) {
        return equipmentRepository.findById(equipmentId)
                .map(equipment -> validateTransition(equipment, targetStatus))
                .orElse(false);
    }
    
    public boolean validateTransition(Equipment equipment, String targetStatus) {
        if (equipment.getEquipmentTypeId() == null) {
            return false;
        }
        
        List<EquipmentStatusTransition> transitions = statusTransitionRepository
                .findByEquipmentTypeIdAndFromStatusCode(
                        equipment.getEquipmentTypeId(), 
                        equipment.getStatusCode());
        
        return equipment.canTransitionTo(targetStatus, transitions);
    }
    
    public boolean transitionToStatus(Long equipmentId, String targetStatus) {
        Optional<Equipment> optEquipment = equipmentRepository.findById(equipmentId);
        if (optEquipment.isEmpty()) {
            return false;
        }
        
        Equipment equipment = optEquipment.get();
        
        if (!validateTransition(equipment, targetStatus)) {
            return false;
        }
        
        Optional<EquipmentStatusConfig> optConfig = statusConfigRepository
                .findByEquipmentTypeIdAndStatusCode(
                        equipment.getEquipmentTypeId(), 
                        targetStatus);
        
        Long configId = optConfig.map(EquipmentStatusConfig::getId).orElse(null);
        equipment.transitionTo(targetStatus, configId);
        equipmentRepository.update(equipment);
        
        return true;
    }
    
    public void startTask(String equipmentCode) {
        equipmentRepository.findByEquipmentCode(equipmentCode).ifPresent(equipment -> {
            equipment.startTask(null);
            equipmentRepository.update(equipment);
        });
    }
    
    public void completeTask(String equipmentCode) {
        equipmentRepository.findByEquipmentCode(equipmentCode).ifPresent(equipment -> {
            equipment.completeTask();
            equipmentRepository.update(equipment);
        });
    }
    
    public void reportBreakdown(String equipmentCode) {
        equipmentRepository.findByEquipmentCode(equipmentCode).ifPresent(equipment -> {
            equipment.reportBreakdown();
            equipmentRepository.update(equipment);
        });
    }
    
    public void repair(String equipmentCode) {
        equipmentRepository.findByEquipmentCode(equipmentCode).ifPresent(equipment -> {
            equipment.repair();
            equipmentRepository.update(equipment);
        });
    }
    
    public void startMaintenance(String equipmentCode) {
        equipmentRepository.findByEquipmentCode(equipmentCode).ifPresent(equipment -> {
            equipment.startMaintenance();
            equipmentRepository.update(equipment);
        });
    }
    
    public void completeMaintenance(String equipmentCode) {
        equipmentRepository.findByEquipmentCode(equipmentCode).ifPresent(equipment -> {
            equipment.completeMaintenance();
            equipmentRepository.update(equipment);
        });
    }
    
    public List<EquipmentStatusConfig> getAvailableStatuses(Long equipmentTypeId) {
        return statusConfigRepository.findByEquipmentTypeIdAndIsActive(equipmentTypeId, true);
    }
    
    public List<String> getNextValidStatuses(Long equipmentTypeId, String currentStatus) {
        return statusTransitionRepository
                .findByEquipmentTypeIdAndFromStatusCode(equipmentTypeId, currentStatus)
                .stream()
                .filter(EquipmentStatusTransition::getIsActive)
                .map(EquipmentStatusTransition::getToStatusCode)
                .toList();
    }
    
    public Optional<String> getDefaultStatus(Long equipmentTypeId) {
        return statusConfigRepository.findByEquipmentTypeIdAndIsInitial(equipmentTypeId, true)
                .map(EquipmentStatusConfig::getStatusCode);
    }
}