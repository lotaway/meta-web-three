package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EquipmentStatusConfig;
import java.util.List;
import java.util.Optional;

public interface EquipmentStatusConfigRepository {
    Optional<EquipmentStatusConfig> findById(Long id);
    List<EquipmentStatusConfig> findByEquipmentTypeId(Long equipmentTypeId);
    List<EquipmentStatusConfig> findByEquipmentTypeIdAndIsActive(Long equipmentTypeId, Boolean isActive);
    Optional<EquipmentStatusConfig> findByEquipmentTypeIdAndStatusCode(Long equipmentTypeId, String statusCode);
    Optional<EquipmentStatusConfig> findByEquipmentTypeIdAndIsInitial(Long equipmentTypeId, Boolean isInitial);
    List<EquipmentStatusConfig> findByStatusCategory(String statusCategory);
    EquipmentStatusConfig save(EquipmentStatusConfig statusConfig);
    void update(EquipmentStatusConfig statusConfig);
    void deleteById(Long id);
}