package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.EquipmentChecklistRecord;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface EquipmentChecklistRecordRepository {
    Optional<EquipmentChecklistRecord> findById(Long id);
    Optional<EquipmentChecklistRecord> findByRecordCode(String recordCode);
    List<EquipmentChecklistRecord> findByEquipmentId(Long equipmentId);
    List<EquipmentChecklistRecord> findByEquipmentCode(String equipmentCode);
    List<EquipmentChecklistRecord> findByTemplateId(Long templateId);
    List<EquipmentChecklistRecord> findByStatus(EquipmentChecklistRecord.RecordStatus status);
    List<EquipmentChecklistRecord> findByCheckerId(String checkerId);
    List<EquipmentChecklistRecord> findOverdueRecords();
    List<EquipmentChecklistRecord> findByEquipmentIdAndCheckPlanTimeBetween(
            Long equipmentId, LocalDateTime startTime, LocalDateTime endTime);
    EquipmentChecklistRecord save(EquipmentChecklistRecord record);
    void update(EquipmentChecklistRecord record);
    void deleteById(Long id);
}