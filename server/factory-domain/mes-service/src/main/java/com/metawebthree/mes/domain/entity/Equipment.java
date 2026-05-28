package com.metawebthree.mes.domain.entity;

import com.metawebthree.mes.domain.entity.EquipmentStatusCode;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class Equipment {
    
    public enum EquipmentStatus {
        IDLE,
        RUNNING,
        BREAKDOWN,
        MAINTENANCE,
        OFFLINE,
        ONLINE,
        WARNING,
        ERROR
    }
    
    private Long id;
    private String equipmentCode;
    private String equipmentName;
    private Long equipmentTypeId;
    private String equipmentTypeCode;
    private String workshopId;
    private String workstationId;
    private String statusCode;
    private EquipmentStatus status;
    private Long statusConfigId;
    private String currentTaskNo;
    private Double utilizationRate;
    private Integer todayOutput;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    private Long totalRunningSeconds;
    private Long totalIdleSeconds;
    private Long totalDowntimeSeconds;
    
    private Map<String, Object> extensionFields;
    
    private String digitalTwinDeviceCode;
    private Double positionX;
    private Double positionY;
    private Double positionZ;
    private Double rotationY;
    private String ipAddress;
    private String macAddress;
    private String mqttTopic;
    private LocalDateTime lastHeartbeat;
    
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(String equipmentCode, String equipmentName, 
                      Long equipmentTypeId, String equipmentTypeCode, String workshopId) {
        this.equipmentCode = equipmentCode;
        this.equipmentName = equipmentName;
        this.equipmentTypeId = equipmentTypeId;
        this.equipmentTypeCode = equipmentTypeCode;
        this.workshopId = workshopId;
        this.statusCode = EquipmentStatusCode.IDLE;
        this.status = EquipmentStatus.IDLE;
        this.utilizationRate = 0.0;
        this.todayOutput = 0;
        this.totalRunningSeconds = 0L;
        this.totalIdleSeconds = 0L;
        this.totalDowntimeSeconds = 0L;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void create(String equipmentCode, String equipmentName, 
                      String equipmentTypeCode, String workshopId) {
        this.equipmentCode = equipmentCode;
        this.equipmentName = equipmentName;
        this.equipmentTypeCode = equipmentTypeCode;
        this.workshopId = workshopId;
        this.statusCode = EquipmentStatusCode.IDLE;
        this.status = EquipmentStatus.IDLE;
        this.utilizationRate = 0.0;
        this.todayOutput = 0;
        this.totalRunningSeconds = 0L;
        this.totalIdleSeconds = 0L;
        this.totalDowntimeSeconds = 0L;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public boolean canTransitionTo(String targetStatus, List<EquipmentStatusTransition> transitions) {
        if (targetStatus == null || this.statusCode == null) {
            return false;
        }
        if (targetStatus.equals(this.statusCode)) {
            return false;
        }
        for (EquipmentStatusTransition transition : transitions) {
            if (transition.canTransition(this.statusCode, null)) {
                if (transition.getToStatusCode().equals(targetStatus)) {
                    return true;
                }
            }
        }
        return false;
    }

    public void transitionTo(String newStatusCode, Long statusConfigId) {
        this.statusCode = newStatusCode;
        this.statusConfigId = statusConfigId;
        this.updatedAt = LocalDateTime.now();
    }

    public void startTask(String taskNo) {
        if (this.status != EquipmentStatus.IDLE) {
            throw new IllegalStateException("只有在 IDLE 状态下才能开始任务，当前状态: " + this.status);
        }
        this.statusCode = EquipmentStatusCode.RUNNING;
        this.status = EquipmentStatus.RUNNING;
        this.currentTaskNo = taskNo;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeTask() {
        if (this.status != EquipmentStatus.RUNNING) {
            throw new IllegalStateException("只有在 RUNNING 状态下才能完成任务，当前状态: " + this.status);
        }
        this.statusCode = EquipmentStatusCode.IDLE;
        this.status = EquipmentStatus.IDLE;
        this.currentTaskNo = null;
        this.todayOutput++;
        this.updatedAt = LocalDateTime.now();
    }

    public void reportBreakdown() {
        if (this.status != EquipmentStatus.RUNNING) {
            throw new IllegalStateException("只有在 RUNNING 状态下才能报故障，当前状态: " + this.status);
        }
        this.statusCode = EquipmentStatusCode.BREAKDOWN;
        this.status = EquipmentStatus.BREAKDOWN;
        this.updatedAt = LocalDateTime.now();
    }

    public void repair() {
        if (this.status != EquipmentStatus.BREAKDOWN) {
            throw new IllegalStateException("只有在 BREAKDOWN 状态下才能维修，当前状态: " + this.status);
        }
        this.statusCode = EquipmentStatusCode.IDLE;
        this.status = EquipmentStatus.IDLE;
        this.updatedAt = LocalDateTime.now();
    }

    public void startMaintenance() {
        if (this.status == EquipmentStatus.RUNNING) {
            throw new IllegalStateException("RUNNING 状态下不能开始保养，当前状态: " + this.status);
        }
        this.statusCode = EquipmentStatusCode.MAINTENANCE;
        this.status = EquipmentStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeMaintenance() {
        if (this.status != EquipmentStatus.MAINTENANCE) {
            throw new IllegalStateException("只有在 MAINTENANCE 状态下才能完成保养，当前状态: " + this.status);
        }
        this.statusCode = EquipmentStatusCode.IDLE;
        this.status = EquipmentStatus.IDLE;
        this.lastMaintenanceTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void heartbeat() {
        this.lastHeartbeat = LocalDateTime.now();
        if (EquipmentStatusCode.OFFLINE.equals(this.statusCode)) {
            this.statusCode = EquipmentStatusCode.ONLINE;
            this.status = EquipmentStatus.ONLINE;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void updatePosition(Double x, Double y, Double z, Double rotation) {
        this.positionX = x;
        this.positionY = y;
        this.positionZ = z;
        this.rotationY = rotation;
        this.updatedAt = LocalDateTime.now();
    }

    public void bindDigitalTwinDevice(String deviceCode) {
        this.digitalTwinDeviceCode = deviceCode;
        this.updatedAt = LocalDateTime.now();
    }

    public void unbindDigitalTwinDevice() {
        this.digitalTwinDeviceCode = null;
        this.updatedAt = LocalDateTime.now();
    }

    public void syncFromDigitalTwin(String dtStatus, LocalDateTime dtLastHeartbeat) {
        if (dtStatus != null) {
            this.statusCode = mapDigitalTwinStatus(dtStatus);
            this.status = mapToEquipmentStatus(dtStatus);
        }
        if (dtLastHeartbeat != null) {
            this.lastHeartbeat = dtLastHeartbeat;
        }
        this.updatedAt = LocalDateTime.now();
    }

    private String mapDigitalTwinStatus(String dtStatus) {
        if (dtStatus == null) return null;
        return switch (dtStatus.toUpperCase()) {
            case "ONLINE" -> EquipmentStatusCode.ONLINE;
            case "OFFLINE" -> EquipmentStatusCode.OFFLINE;
            case "RUNNING" -> EquipmentStatusCode.RUNNING;
            case "IDLE" -> EquipmentStatusCode.IDLE;
            case "WARNING" -> EquipmentStatusCode.WARNING;
            case "ERROR" -> EquipmentStatusCode.ERROR;
            case "MAINTENANCE" -> EquipmentStatusCode.MAINTENANCE;
            default -> dtStatus.toUpperCase();
        };
    }

    private EquipmentStatus mapToEquipmentStatus(String dtStatus) {
        if (dtStatus == null) return null;
        return switch (dtStatus.toUpperCase()) {
            case "ONLINE" -> EquipmentStatus.ONLINE;
            case "OFFLINE" -> EquipmentStatus.OFFLINE;
            case "RUNNING" -> EquipmentStatus.RUNNING;
            case "IDLE" -> EquipmentStatus.IDLE;
            case "WARNING" -> EquipmentStatus.WARNING;
            case "ERROR" -> EquipmentStatus.ERROR;
            case "MAINTENANCE" -> EquipmentStatus.MAINTENANCE;
            default -> null;
        };
    }

    public void updateExtensionField(String fieldName, Object value) {
        if (this.extensionFields == null) {
            this.extensionFields = new java.util.HashMap<>();
        }
        this.extensionFields.put(fieldName, value);
        this.updatedAt = LocalDateTime.now();
    }

    public Object getExtensionField(String fieldName) {
        if (this.extensionFields == null) {
            return null;
        }
        return this.extensionFields.get(fieldName);
    }

    public void calculateOEE(Integer plannedProductionTime, Integer idealCycleTime, Integer goodProductCount) {
        if (plannedProductionTime == null || plannedProductionTime == 0) {
            this.utilizationRate = 0.0;
            return;
        }
        if (totalRunningSeconds == null || totalRunningSeconds == 0) {
            this.utilizationRate = 0.0;
            return;
        }
        Double availability = (double) (totalRunningSeconds + totalIdleSeconds) / plannedProductionTime;
        Double performance = idealCycleTime != null && todayOutput != null && todayOutput > 0 
            ? (double) (todayOutput * idealCycleTime) / (totalRunningSeconds * 60) 
            : 1.0;
        Double quality = goodProductCount != null && todayOutput != null && todayOutput > 0
            ? (double) goodProductCount / todayOutput
            : 1.0;
        this.utilizationRate = availability * performance * quality * 100;
    }

    public void bindWorkstation(String workstationId) {
        this.workstationId = workstationId;
        this.updatedAt = LocalDateTime.now();
    }

    public void unbindWorkstation() {
        this.workstationId = null;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEquipmentCode() { return equipmentCode; }
    public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
    public String getEquipmentName() { return equipmentName; }
    public void setEquipmentName(String equipmentName) { this.equipmentName = equipmentName; }
    public Long getEquipmentTypeId() { return equipmentTypeId; }
    public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
    public String getEquipmentTypeCode() { return equipmentTypeCode; }
    public void setEquipmentTypeCode(String equipmentTypeCode) { this.equipmentTypeCode = equipmentTypeCode; }
    public String getEquipmentType() { return equipmentTypeCode; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getWorkstationId() { return workstationId; }
    public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
    public String getStatusCode() { return statusCode; }
    public void setStatusCode(String statusCode) { this.statusCode = statusCode; }
    public EquipmentStatus getStatus() { return status; }
    public void setStatus(EquipmentStatus status) { this.status = status; }
    public Long getStatusConfigId() { return statusConfigId; }
    public void setStatusConfigId(Long statusConfigId) { this.statusConfigId = statusConfigId; }
    public String getCurrentTaskNo() { return currentTaskNo; }
    public void setCurrentTaskNo(String currentTaskNo) { this.currentTaskNo = currentTaskNo; }
    public Double getUtilizationRate() { return utilizationRate; }
    public void setUtilizationRate(Double utilizationRate) { this.utilizationRate = utilizationRate; }
    public Integer getTodayOutput() { return todayOutput; }
    public void setTodayOutput(Integer todayOutput) { this.todayOutput = todayOutput; }
    public LocalDateTime getLastMaintenanceTime() { return lastMaintenanceTime; }
    public void setLastMaintenanceTime(LocalDateTime lastMaintenanceTime) { this.lastMaintenanceTime = lastMaintenanceTime; }
    public LocalDateTime getNextMaintenanceTime() { return nextMaintenanceTime; }
    public void setNextMaintenanceTime(LocalDateTime nextMaintenanceTime) { this.nextMaintenanceTime = nextMaintenanceTime; }
    public Long getTotalRunningSeconds() { return totalRunningSeconds; }
    public void setTotalRunningSeconds(Long totalRunningSeconds) { this.totalRunningSeconds = totalRunningSeconds; }
    public Long getTotalIdleSeconds() { return totalIdleSeconds; }
    public void setTotalIdleSeconds(Long totalIdleSeconds) { this.totalIdleSeconds = totalIdleSeconds; }
    public Long getTotalDowntimeSeconds() { return totalDowntimeSeconds; }
    public void setTotalDowntimeSeconds(Long totalDowntimeSeconds) { this.totalDowntimeSeconds = totalDowntimeSeconds; }
    public Map<String, Object> getExtensionFields() { return extensionFields; }
    public void setExtensionFields(Map<String, Object> extensionFields) { this.extensionFields = extensionFields; }
    public String getDigitalTwinDeviceCode() { return digitalTwinDeviceCode; }
    public void setDigitalTwinDeviceCode(String digitalTwinDeviceCode) { this.digitalTwinDeviceCode = digitalTwinDeviceCode; }
    public Double getPositionX() { return positionX; }
    public void setPositionX(Double positionX) { this.positionX = positionX; }
    public Double getPositionY() { return positionY; }
    public void setPositionY(Double positionY) { this.positionY = positionY; }
    public Double getPositionZ() { return positionZ; }
    public void setPositionZ(Double positionZ) { this.positionZ = positionZ; }
    public Double getRotationY() { return rotationY; }
    public void setRotationY(Double rotationY) { this.rotationY = rotationY; }
    public String getIpAddress() { return ipAddress; }
    public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
    public String getMacAddress() { return macAddress; }
    public void setMacAddress(String macAddress) { this.macAddress = macAddress; }
    public String getMqttTopic() { return mqttTopic; }
    public void setMqttTopic(String mqttTopic) { this.mqttTopic = mqttTopic; }
    public LocalDateTime getLastHeartbeat() { return lastHeartbeat; }
    public void setLastHeartbeat(LocalDateTime lastHeartbeat) { this.lastHeartbeat = lastHeartbeat; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}