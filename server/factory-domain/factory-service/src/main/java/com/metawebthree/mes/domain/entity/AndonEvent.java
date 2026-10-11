package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class AndonEvent {
    
    private Long id;
    private String eventNo;
    private Long andonTypeId;
    private String andonTypeCode;
    private String andonTypeName;
    private Long andonLevelId;
    private String levelCode;
    private String levelName;
    private String triggerMethod;
    private String sourceSystem;
    private String sourceId;
    private String workshopId;
    private Long workstationId;
    private String equipmentId;
    private String productCode;
    private String workOrderNo;
    private String description;
    private String photoUrl;
    private String reporterId;
    private String reporterName;
    private AndonEventStatus status;
    private String currentHandlerId;
    private String currentHandlerName;
    private LocalDateTime occurredAt;
    private LocalDateTime acknowledgedAt;
    private LocalDateTime resolvedAt;
    private Integer escalationCount;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum AndonEventStatus {
        PENDING,
        ACKNOWLEDGED,
        HANDLING,
        RESOLVED,
        CLOSED,
        ESCALATED
    }
    
    public enum TriggerMethod {
        MANUAL_BUTTON,
        AUTO_DETECTION,
        SCAN_QR,
        API_CALL
    }
    
    public static AndonEvent create(Long andonTypeId, String andonTypeCode, String andonTypeName,
                                    Long andonLevelId, String levelCode, String levelName,
                                    String triggerMethod, String workshopId, Long workstationId,
                                    String equipmentId, String reporterId, String reporterName,
                                    String description) {
        AndonEvent event = new AndonEvent();
        event.eventNo = generateEventNo();
        event.andonTypeId = andonTypeId;
        event.andonTypeCode = andonTypeCode;
        event.andonTypeName = andonTypeName;
        event.andonLevelId = andonLevelId;
        event.levelCode = levelCode;
        event.levelName = levelName;
        event.triggerMethod = triggerMethod;
        event.workshopId = workshopId;
        event.workstationId = workstationId;
        event.equipmentId = equipmentId;
        event.reporterId = reporterId;
        event.reporterName = reporterName;
        event.description = description;
        event.status = AndonEventStatus.PENDING;
        event.escalationCount = 0;
        event.occurredAt = LocalDateTime.now();
        event.createdAt = LocalDateTime.now();
        event.updatedAt = LocalDateTime.now();
        return event;
    }
    
    private static String generateEventNo() {
        return "ANDON-" + System.currentTimeMillis();
    }
    
    public void acknowledge(String handlerId, String handlerName) {
        this.status = AndonEventStatus.ACKNOWLEDGED;
        this.currentHandlerId = handlerId;
        this.currentHandlerName = handlerName;
        this.acknowledgedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void startHandling() {
        this.status = AndonEventStatus.HANDLING;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void resolve() {
        this.status = AndonEventStatus.RESOLVED;
        this.resolvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void close() {
        this.status = AndonEventStatus.CLOSED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void escalate() {
        this.status = AndonEventStatus.ESCALATED;
        this.escalationCount++;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void addPhoto(String photoUrl) {
        this.photoUrl = photoUrl;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void assignHandler(String handlerId, String handlerName) {
        this.currentHandlerId = handlerId;
        this.currentHandlerName = handlerName;
        this.updatedAt = LocalDateTime.now();
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEventNo() { return eventNo; }
    public void setEventNo(String eventNo) { this.eventNo = eventNo; }
    public Long getAndonTypeId() { return andonTypeId; }
    public void setAndonTypeId(Long andonTypeId) { this.andonTypeId = andonTypeId; }
    public String getAndonTypeCode() { return andonTypeCode; }
    public void setAndonTypeCode(String andonTypeCode) { this.andonTypeCode = andonTypeCode; }
    public String getAndonTypeName() { return andonTypeName; }
    public void setAndonTypeName(String andonTypeName) { this.andonTypeName = andonTypeName; }
    public Long getAndonLevelId() { return andonLevelId; }
    public void setAndonLevelId(Long andonLevelId) { this.andonLevelId = andonLevelId; }
    public String getLevelCode() { return levelCode; }
    public void setLevelCode(String levelCode) { this.levelCode = levelCode; }
    public String getLevelName() { return levelName; }
    public void setLevelName(String levelName) { this.levelName = levelName; }
    public String getTriggerMethod() { return triggerMethod; }
    public void setTriggerMethod(String triggerMethod) { this.triggerMethod = triggerMethod; }
    public String getSourceSystem() { return sourceSystem; }
    public void setSourceSystem(String sourceSystem) { this.sourceSystem = sourceSystem; }
    public String getSourceId() { return sourceId; }
    public void setSourceId(String sourceId) { this.sourceId = sourceId; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public Long getWorkstationId() { return workstationId; }
    public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
    public String getEquipmentId() { return equipmentId; }
    public void setEquipmentId(String equipmentId) { this.equipmentId = equipmentId; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getWorkOrderNo() { return workOrderNo; }
    public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getPhotoUrl() { return photoUrl; }
    public void setPhotoUrl(String photoUrl) { this.photoUrl = photoUrl; }
    public String getReporterId() { return reporterId; }
    public void setReporterId(String reporterId) { this.reporterId = reporterId; }
    public String getReporterName() { return reporterName; }
    public void setReporterName(String reporterName) { this.reporterName = reporterName; }
    public AndonEventStatus getStatus() { return status; }
    public void setStatus(AndonEventStatus status) { this.status = status; }
    public String getCurrentHandlerId() { return currentHandlerId; }
    public void setCurrentHandlerId(String currentHandlerId) { this.currentHandlerId = currentHandlerId; }
    public String getCurrentHandlerName() { return currentHandlerName; }
    public void setCurrentHandlerName(String currentHandlerName) { this.currentHandlerName = currentHandlerName; }
    public LocalDateTime getOccurredAt() { return occurredAt; }
    public void setOccurredAt(LocalDateTime occurredAt) { this.occurredAt = occurredAt; }
    public LocalDateTime getAcknowledgedAt() { return acknowledgedAt; }
    public void setAcknowledgedAt(LocalDateTime acknowledgedAt) { this.acknowledgedAt = acknowledgedAt; }
    public LocalDateTime getResolvedAt() { return resolvedAt; }
    public void setResolvedAt(LocalDateTime resolvedAt) { this.resolvedAt = resolvedAt; }
    public Integer getEscalationCount() { return escalationCount; }
    public void setEscalationCount(Integer escalationCount) { this.escalationCount = escalationCount; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}