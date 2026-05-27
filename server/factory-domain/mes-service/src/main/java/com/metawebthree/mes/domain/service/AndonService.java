package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.AndonType;
import com.metawebthree.mes.domain.entity.AndonLevel;
import com.metawebthree.mes.domain.entity.AndonEvent;
import java.util.List;
import java.util.Optional;

public interface AndonService {
    
    AndonType createAndonType(String typeCode, String typeName, AndonType.AndonCategory category);
    void updateAndonType(Long id, String typeName, AndonType.AndonCategory category, String description,
                        Boolean requirePhoto, Boolean requireConfirm, Integer defaultEscalationMinutes,
                        String defaultProcessTemplate);
    void activateAndonType(Long id);
    void deactivateAndonType(Long id);
    Optional<AndonType> getAndonTypeById(Long id);
    Optional<AndonType> getAndonTypeByCode(String typeCode);
    List<AndonType> getAndonTypesByCategory(AndonType.AndonCategory category);
    List<AndonType> getAllActiveAndonTypes();
    
    AndonLevel createAndonLevel(String levelCode, String levelName, Integer levelValue);
    void updateAndonLevel(Long id, String levelName, Integer levelValue, Integer responseTimeoutMinutes,
                         Integer handlingTimeoutMinutes, String colorCode, String description);
    void activateAndonLevel(Long id);
    void deactivateAndonLevel(Long id);
    Optional<AndonLevel> getAndonLevelById(Long id);
    Optional<AndonLevel> getAndonLevelByCode(String levelCode);
    List<AndonLevel> getAllActiveAndonLevels();
    
    AndonEvent triggerAndon(Long andonTypeId, Long andonLevelId, String triggerMethod,
                          String workshopId, String workstationId, String equipmentId,
                          String reporterId, String reporterName, String description);
    void acknowledgeAndonEvent(Long eventId, String handlerId, String handlerName);
    void startHandling(Long eventId);
    void resolveAndonEvent(Long eventId);
    void closeAndonEvent(Long eventId);
    void escalateAndonEvent(Long eventId);
    void addPhotoToEvent(Long eventId, String photoUrl);
    void assignHandler(Long eventId, String handlerId, String handlerName);
    Optional<AndonEvent> getAndonEventById(Long id);
    Optional<AndonEvent> getAndonEventByNo(String eventNo);
    List<AndonEvent> getAndonEventsByStatus(AndonEvent.AndonEventStatus status);
    List<AndonEvent> getAndonEventsByType(Long andonTypeId);
    List<AndonEvent> getAndonEventsByWorkshop(String workshopId);
    List<AndonEvent> getUnhandledEventsOlderThan(int minutes);
}