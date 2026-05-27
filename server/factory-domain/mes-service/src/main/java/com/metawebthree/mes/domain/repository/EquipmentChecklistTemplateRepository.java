package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EquipmentChecklistTemplate;
import java.util.List;
import java.util.Optional;

public interface EquipmentChecklistTemplateRepository {
    Optional<EquipmentChecklistTemplate> findById(Long id);
    Optional<EquipmentChecklistTemplate> findByTemplateCode(String templateCode);
    List<EquipmentChecklistTemplate> findByEquipmentTypeCode(String equipmentTypeCode);
    List<EquipmentChecklistTemplate> findByStatus(EquipmentChecklistTemplate.TemplateStatus status);
    List<EquipmentChecklistTemplate> findAll();
    EquipmentChecklistTemplate save(EquipmentChecklistTemplate template);
    void update(EquipmentChecklistTemplate template);
    void deleteById(Long id);
}