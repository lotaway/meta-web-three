package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.AndonEvent;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface AndonEventRepository {
    Optional<AndonEvent> findById(Long id);
    Optional<AndonEvent> findByEventNo(String eventNo);
    List<AndonEvent> findByStatus(AndonEvent.AndonEventStatus status);
    List<AndonEvent> findByAndonTypeId(Long andonTypeId);
    List<AndonEvent> findByAndonLevelId(Long andonLevelId);
    List<AndonEvent> findByWorkshopId(String workshopId);
    List<AndonEvent> findByEquipmentId(String equipmentId);
    List<AndonEvent> findByReporterId(String reporterId);
    List<AndonEvent> findByCurrentHandlerId(String currentHandlerId);
    List<AndonEvent> findByStatusAndOccurredAtBefore(AndonEvent.AndonEventStatus status, LocalDateTime time);
    List<AndonEvent> findAll();
    AndonEvent save(AndonEvent andonEvent);
    void update(AndonEvent andonEvent);
    void deleteById(Long id);
}