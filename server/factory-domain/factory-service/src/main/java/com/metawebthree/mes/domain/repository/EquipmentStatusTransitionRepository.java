package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EquipmentStatusTransition;
import java.util.List;
import java.util.Optional;

public interface EquipmentStatusTransitionRepository {
    Optional<EquipmentStatusTransition> findById(Long id);
    List<EquipmentStatusTransition> findByEquipmentTypeId(Long equipmentTypeId);
    List<EquipmentStatusTransition> findByEquipmentTypeIdAndIsActive(Long equipmentTypeId, Boolean isActive);
    List<EquipmentStatusTransition> findByEquipmentTypeIdAndFromStatusCode(Long equipmentTypeId, String fromStatusCode);
    Optional<EquipmentStatusTransition> findByEquipmentTypeIdAndFromStatusCodeAndToStatusCode(
            Long equipmentTypeId, String fromStatusCode, String toStatusCode);
    EquipmentStatusTransition save(EquipmentStatusTransition transition);
    void update(EquipmentStatusTransition transition);
    void deleteById(Long id);
}