package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EquipmentType;
import java.util.List;
import java.util.Optional;

public interface EquipmentTypeRepository {
    Optional<EquipmentType> findById(Long id);
    Optional<EquipmentType> findByTypeCode(String typeCode);
    List<EquipmentType> findByCategory(String category);
    List<EquipmentType> findByIsActive(Boolean isActive);
    List<EquipmentType> findAll();
    EquipmentType save(EquipmentType equipmentType);
    void update(EquipmentType equipmentType);
    void deleteById(Long id);
}