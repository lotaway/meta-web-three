package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.Equipment;
import java.util.List;
import java.util.Optional;

public interface EquipmentRepository {
    Optional<Equipment> findById(Long id);
    Optional<Equipment> findByEquipmentCode(String equipmentCode);
    List<Equipment> findByWorkshopId(String workshopId);
    List<Equipment> findByStatus(Equipment.EquipmentStatus status);
    List<Equipment> findByWorkstationId(String workstationId);
    Equipment save(Equipment equipment);
    void update(Equipment equipment);
    void deleteById(Long id);
}