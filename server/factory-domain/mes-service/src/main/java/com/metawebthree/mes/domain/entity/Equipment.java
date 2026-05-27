package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class Equipment {
    private Long id;
    private String equipmentCode;
    private String equipmentName;
    private String equipmentType;
    private String workshopId;
    private String workstationId;
    private EquipmentStatus status;
    private Double utilizationRate;
    private Integer todayOutput;
    private String currentTaskNo;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    
    // 数字孪生关联字段
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

    public enum EquipmentStatus {
        IDLE, RUNNING, MAINTENANCE, BREAKDOWN, SCRAP,
        ONLINE, OFFLINE, WARNING, ERROR
    }

    public void create(String equipmentCode, String equipmentName, 
                      String equipmentType, String workshopId) {
        this.equipmentCode = equipmentCode;
        this.equipmentName = equipmentName;
        this.equipmentType = equipmentType;
        this.workshopId = workshopId;
        this.status = EquipmentStatus.IDLE;
        this.utilizationRate = 0.0;
        this.todayOutput = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void startTask(String taskNo) {
        if (status != EquipmentStatus.IDLE) {
            throw new IllegalStateException("Equipment is not idle");
        }
        this.status = EquipmentStatus.RUNNING;
        this.currentTaskNo = taskNo;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeTask() {
        if (status != EquipmentStatus.RUNNING) {
            throw new IllegalStateException("Equipment is not running");
        }
        this.status = EquipmentStatus.IDLE;
        this.currentTaskNo = null;
        this.todayOutput++;
        this.updatedAt = LocalDateTime.now();
    }

    public void reportBreakdown() {
        if (status != EquipmentStatus.RUNNING) {
            throw new IllegalStateException("Equipment is not running");
        }
        this.status = EquipmentStatus.BREAKDOWN;
        this.updatedAt = LocalDateTime.now();
    }

    public void repair() {
        if (status != EquipmentStatus.BREAKDOWN) {
            throw new IllegalStateException("Equipment is not broken down");
        }
        this.status = EquipmentStatus.IDLE;
        this.updatedAt = LocalDateTime.now();
    }

    public void startMaintenance() {
        if (status == EquipmentStatus.RUNNING) {
            throw new IllegalStateException("Cannot maintain while running");
        }
        this.status = EquipmentStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeMaintenance() {
        if (status != EquipmentStatus.MAINTENANCE) {
            throw new IllegalStateException("Equipment is not in maintenance");
        }
        this.status = EquipmentStatus.IDLE;
        this.lastMaintenanceTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void heartbeat() {
        this.lastHeartbeat = LocalDateTime.now();
        if (this.status == EquipmentStatus.OFFLINE) {
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
            this.status = fromDigitalTwinStatusString(dtStatus);
        }
        if (dtLastHeartbeat != null) {
            this.lastHeartbeat = dtLastHeartbeat;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public static EquipmentStatus fromDigitalTwinStatusString(String dtStatus) {
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

    public String toDigitalTwinStatusString() {
        if (this.status == null) return null;
        return switch (this.status) {
            case ONLINE -> "ONLINE";
            case OFFLINE -> "OFFLINE";
            case RUNNING -> "RUNNING";
            case IDLE -> "IDLE";
            case WARNING -> "WARNING";
            case ERROR -> "ERROR";
            case MAINTENANCE -> "MAINTENANCE";
            case BREAKDOWN -> "ERROR";
            case SCRAP -> "OFFLINE";
        };
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getEquipmentCode() { return equipmentCode; }
    public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
    public String getEquipmentName() { return equipmentName; }
    public void setEquipmentName(String equipmentName) { this.equipmentName = equipmentName; }
    public String getEquipmentType() { return equipmentType; }
    public void setEquipmentType(String equipmentType) { this.equipmentType = equipmentType; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getWorkstationId() { return workstationId; }
    public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
    public EquipmentStatus getStatus() { return status; }
    public void setStatus(EquipmentStatus status) { this.status = status; }
    public Double getUtilizationRate() { return utilizationRate; }
    public void setUtilizationRate(Double utilizationRate) { this.utilizationRate = utilizationRate; }
    public Integer getTodayOutput() { return todayOutput; }
    public void setTodayOutput(Integer todayOutput) { this.todayOutput = todayOutput; }
    public String getCurrentTaskNo() { return currentTaskNo; }
    public void setCurrentTaskNo(String currentTaskNo) { this.currentTaskNo = currentTaskNo; }
    public LocalDateTime getLastMaintenanceTime() { return lastMaintenanceTime; }
    public void setLastMaintenanceTime(LocalDateTime lastMaintenanceTime) { this.lastMaintenanceTime = lastMaintenanceTime; }
    public LocalDateTime getNextMaintenanceTime() { return nextMaintenanceTime; }
    public void setNextMaintenanceTime(LocalDateTime nextMaintenanceTime) { this.nextMaintenanceTime = nextMaintenanceTime; }
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