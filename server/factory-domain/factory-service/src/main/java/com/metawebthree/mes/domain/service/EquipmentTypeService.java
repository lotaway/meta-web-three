package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.EquipmentStatusConfig;
import com.metawebthree.mes.domain.entity.EquipmentStatusTransition;
import com.metawebthree.mes.domain.entity.EquipmentType;
import com.metawebthree.mes.domain.repository.EquipmentStatusConfigRepository;
import com.metawebthree.mes.domain.repository.EquipmentStatusTransitionRepository;
import com.metawebthree.mes.domain.repository.EquipmentTypeRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class EquipmentTypeService {
    
    private final EquipmentTypeRepository equipmentTypeRepository;
    private final EquipmentStatusConfigRepository statusConfigRepository;
    private final EquipmentStatusTransitionRepository statusTransitionRepository;
    
    public EquipmentTypeService(
            EquipmentTypeRepository equipmentTypeRepository,
            EquipmentStatusConfigRepository statusConfigRepository,
            EquipmentStatusTransitionRepository statusTransitionRepository) {
        this.equipmentTypeRepository = equipmentTypeRepository;
        this.statusConfigRepository = statusConfigRepository;
        this.statusTransitionRepository = statusTransitionRepository;
    }
    
    public EquipmentType createType(String typeCode, String typeName, String category) {
        EquipmentType type = new EquipmentType();
        type.create(typeCode, typeName, category);
        return equipmentTypeRepository.save(type);
    }
    
    public Optional<EquipmentType> getTypeById(Long id) {
        return equipmentTypeRepository.findById(id);
    }
    
    public Optional<EquipmentType> getTypeByCode(String typeCode) {
        return equipmentTypeRepository.findByTypeCode(typeCode);
    }
    
    public List<EquipmentType> getAllTypes() {
        return equipmentTypeRepository.findAll();
    }
    
    public List<EquipmentType> getTypesByCategory(String category) {
        return equipmentTypeRepository.findByCategory(category);
    }
    
    public List<EquipmentType> getActiveTypes() {
        return equipmentTypeRepository.findByIsActive(true);
    }
    
    public EquipmentType updateType(EquipmentType type) {
        equipmentTypeRepository.update(type);
        return type;
    }
    
    public void deleteType(Long id) {
        equipmentTypeRepository.deleteById(id);
    }
    
    public void addAttribute(Long typeId, EquipmentType.EquipmentTypeAttribute attr) {
        equipmentTypeRepository.findById(typeId).ifPresent(type -> {
            type.addAttribute(attr);
            equipmentTypeRepository.update(type);
        });
    }
    
    public void removeAttribute(Long typeId, String attrCode) {
        equipmentTypeRepository.findById(typeId).ifPresent(type -> {
            type.removeAttribute(attrCode);
            equipmentTypeRepository.update(type);
        });
    }
    
    public void bindStatusMachine(Long typeId, Long machineId) {
        equipmentTypeRepository.findById(typeId).ifPresent(type -> {
            type.bindStatusMachine(machineId);
            equipmentTypeRepository.update(type);
        });
    }
    
    public EquipmentStatusConfig addStatusConfig(Long typeId, String statusCode, String statusName, String statusCategory) {
        EquipmentStatusConfig config = new EquipmentStatusConfig();
        config.create(typeId, statusCode, statusName, statusCategory);
        return statusConfigRepository.save(config);
    }
    
    public List<EquipmentStatusConfig> getStatusConfigs(Long typeId) {
        return statusConfigRepository.findByEquipmentTypeId(typeId);
    }
    
    public Optional<EquipmentStatusConfig> getStatusConfig(Long typeId, String statusCode) {
        return statusConfigRepository.findByEquipmentTypeIdAndStatusCode(typeId, statusCode);
    }
    
    public Optional<EquipmentStatusConfig> getInitialStatus(Long typeId) {
        return statusConfigRepository.findByEquipmentTypeIdAndIsInitial(typeId, true);
    }
    
    public EquipmentStatusTransition addStatusTransition(
            Long typeId, String fromStatus, String toStatus, String action) {
        EquipmentStatusTransition transition = new EquipmentStatusTransition();
        transition.create(typeId, fromStatus, toStatus, action);
        return statusTransitionRepository.save(transition);
    }
    
    public List<EquipmentStatusTransition> getStatusTransitions(Long typeId) {
        return statusTransitionRepository.findByEquipmentTypeId(typeId);
    }
    
    public List<EquipmentStatusTransition> getTransitionsFromStatus(Long typeId, String fromStatus) {
        return statusTransitionRepository.findByEquipmentTypeIdAndFromStatusCode(typeId, fromStatus);
    }
    
    public boolean canTransition(Long typeId, String fromStatus, String toStatus) {
        return statusTransitionRepository
                .findByEquipmentTypeIdAndFromStatusCodeAndToStatusCode(typeId, fromStatus, toStatus)
                .map(EquipmentStatusTransition::getIsActive)
                .orElse(false);
    }
}