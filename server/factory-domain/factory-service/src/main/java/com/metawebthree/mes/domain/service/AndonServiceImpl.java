package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.AndonType;
import com.metawebthree.mes.domain.entity.AndonLevel;
import com.metawebthree.mes.domain.entity.AndonEvent;
import com.metawebthree.mes.domain.repository.AndonTypeRepository;
import com.metawebthree.mes.domain.repository.AndonLevelRepository;
import com.metawebthree.mes.domain.repository.AndonEventRepository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public class AndonServiceImpl implements AndonService {
    
    private final AndonTypeRepository andonTypeRepository;
    private final AndonLevelRepository andonLevelRepository;
    private final AndonEventRepository andonEventRepository;
    
    public AndonServiceImpl(AndonTypeRepository andonTypeRepository,
                           AndonLevelRepository andonLevelRepository,
                           AndonEventRepository andonEventRepository) {
        this.andonTypeRepository = andonTypeRepository;
        this.andonLevelRepository = andonLevelRepository;
        this.andonEventRepository = andonEventRepository;
    }
    
    @Override
    public AndonType createAndonType(String typeCode, String typeName, AndonType.AndonCategory category) {
        if (andonTypeRepository.findByTypeCode(typeCode).isPresent()) {
            throw new IllegalArgumentException("AndonType with code " + typeCode + " already exists");
        }
        AndonType type = AndonType.create(typeCode, typeName, category);
        return andonTypeRepository.save(type);
    }
    
    @Override
    public void updateAndonType(Long id, String typeName, AndonType.AndonCategory category, String description,
                               Boolean requirePhoto, Boolean requireConfirm, Integer defaultEscalationMinutes,
                               String defaultProcessTemplate) {
        AndonType type = andonTypeRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("AndonType not found: " + id));
        type.update(typeName, category, description, requirePhoto, requireConfirm, 
                   defaultEscalationMinutes, defaultProcessTemplate);
        andonTypeRepository.update(type);
    }
    
    @Override
    public void activateAndonType(Long id) {
        AndonType type = andonTypeRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("AndonType not found: " + id));
        type.activate();
        andonTypeRepository.update(type);
    }
    
    @Override
    public void deactivateAndonType(Long id) {
        AndonType type = andonTypeRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("AndonType not found: " + id));
        type.deactivate();
        andonTypeRepository.update(type);
    }
    
    @Override
    public Optional<AndonType> getAndonTypeById(Long id) {
        return andonTypeRepository.findById(id);
    }
    
    @Override
    public Optional<AndonType> getAndonTypeByCode(String typeCode) {
        return andonTypeRepository.findByTypeCode(typeCode);
    }
    
    @Override
    public List<AndonType> getAndonTypesByCategory(AndonType.AndonCategory category) {
        return andonTypeRepository.findByCategory(category);
    }
    
    @Override
    public List<AndonType> getAllActiveAndonTypes() {
        return andonTypeRepository.findByStatus(AndonType.AndonStatus.ACTIVE);
    }
    
    @Override
    public AndonLevel createAndonLevel(String levelCode, String levelName, Integer levelValue) {
        if (andonLevelRepository.findByLevelCode(levelCode).isPresent()) {
            throw new IllegalArgumentException("AndonLevel with code " + levelCode + " already exists");
        }
        if (andonLevelRepository.findByLevelValue(levelValue).isPresent()) {
            throw new IllegalArgumentException("AndonLevel with value " + levelValue + " already exists");
        }
        AndonLevel level = AndonLevel.create(levelCode, levelName, levelValue);
        return andonLevelRepository.save(level);
    }
    
    @Override
    public void updateAndonLevel(Long id, String levelName, Integer levelValue, Integer responseTimeoutMinutes,
                                 Integer handlingTimeoutMinutes, String colorCode, String description) {
        AndonLevel level = andonLevelRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("AndonLevel not found: " + id));
        level.update(levelName, levelValue, responseTimeoutMinutes, handlingTimeoutMinutes, colorCode, description);
        andonLevelRepository.update(level);
    }
    
    @Override
    public void activateAndonLevel(Long id) {
        AndonLevel level = andonLevelRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("AndonLevel not found: " + id));
        level.activate();
        andonLevelRepository.update(level);
    }
    
    @Override
    public void deactivateAndonLevel(Long id) {
        AndonLevel level = andonLevelRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("AndonLevel not found: " + id));
        level.deactivate();
        andonLevelRepository.update(level);
    }
    
    @Override
    public Optional<AndonLevel> getAndonLevelById(Long id) {
        return andonLevelRepository.findById(id);
    }
    
    @Override
    public Optional<AndonLevel> getAndonLevelByCode(String levelCode) {
        return andonLevelRepository.findByLevelCode(levelCode);
    }
    
    @Override
    public List<AndonLevel> getAllActiveAndonLevels() {
        return andonLevelRepository.findByStatus(AndonLevel.AndonLevelStatus.ACTIVE);
    }
    
    @Override
    public AndonEvent triggerAndon(Long andonTypeId, Long andonLevelId, String triggerMethod,
                                  String workshopId, String workstationId, String equipmentId,
                                  String reporterId, String reporterName, String description) {
        AndonType andonType = andonTypeRepository.findById(andonTypeId)
            .orElseThrow(() -> new IllegalArgumentException("AndonType not found: " + andonTypeId));
        if (!andonType.isActive()) {
            throw new IllegalStateException("AndonType is not active: " + andonTypeId);
        }
        
        AndonLevel andonLevel = andonLevelRepository.findById(andonLevelId)
            .orElseThrow(() -> new IllegalArgumentException("AndonLevel not found: " + andonLevelId));
        
        AndonEvent event = AndonEvent.create(
            andonTypeId, andonType.getTypeCode(), andonType.getTypeName(),
            andonLevelId, andonLevel.getLevelCode(), andonLevel.getLevelName(),
            triggerMethod, workshopId, workstationId != null ? Long.parseLong(workstationId) : null, equipmentId,
            reporterId, reporterName, description
        );
        
        return andonEventRepository.save(event);
    }
    
    @Override
    public void acknowledgeAndonEvent(Long eventId, String handlerId, String handlerName) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.acknowledge(handlerId, handlerName);
        andonEventRepository.update(event);
    }
    
    @Override
    public void startHandling(Long eventId) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.startHandling();
        andonEventRepository.update(event);
    }
    
    @Override
    public void resolveAndonEvent(Long eventId) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.resolve();
        andonEventRepository.update(event);
    }
    
    @Override
    public void closeAndonEvent(Long eventId) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.close();
        andonEventRepository.update(event);
    }
    
    @Override
    public void escalateAndonEvent(Long eventId) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.escalate();
        andonEventRepository.update(event);
    }
    
    @Override
    public void addPhotoToEvent(Long eventId, String photoUrl) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.addPhoto(photoUrl);
        andonEventRepository.update(event);
    }
    
    @Override
    public void assignHandler(Long eventId, String handlerId, String handlerName) {
        AndonEvent event = andonEventRepository.findById(eventId)
            .orElseThrow(() -> new IllegalArgumentException("AndonEvent not found: " + eventId));
        event.assignHandler(handlerId, handlerName);
        andonEventRepository.update(event);
    }
    
    @Override
    public Optional<AndonEvent> getAndonEventById(Long id) {
        return andonEventRepository.findById(id);
    }
    
    @Override
    public Optional<AndonEvent> getAndonEventByNo(String eventNo) {
        return andonEventRepository.findByEventNo(eventNo);
    }
    
    @Override
    public List<AndonEvent> getAndonEventsByStatus(AndonEvent.AndonEventStatus status) {
        return andonEventRepository.findByStatus(status);
    }
    
    @Override
    public List<AndonEvent> getAndonEventsByType(Long andonTypeId) {
        return andonEventRepository.findByAndonTypeId(andonTypeId);
    }
    
    @Override
    public List<AndonEvent> getAndonEventsByWorkshop(String workshopId) {
        return andonEventRepository.findByWorkshopId(workshopId);
    }
    
    @Override
    public List<AndonEvent> getUnhandledEventsOlderThan(int minutes) {
        LocalDateTime threshold = LocalDateTime.now().minusMinutes(minutes);
        return andonEventRepository.findByStatusAndOccurredAtBefore(AndonEvent.AndonEventStatus.PENDING, threshold);
    }
}