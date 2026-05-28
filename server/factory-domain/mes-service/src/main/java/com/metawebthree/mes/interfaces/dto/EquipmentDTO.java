package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import com.metawebthree.mes.domain.entity.Equipment;

public class EquipmentDTO {
    
    private Long id;
    private String equipmentCode;
    private String equipmentName;
    private Long equipmentTypeId;
    private String equipmentTypeCode;
    private String workshopId;
    private String workstationId;
    private String statusCode;
    private String status;
    private Long statusConfigId;
    private String currentTaskNo;
    private Double utilizationRate;
    private Integer todayOutput;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    private Long totalRunningSeconds;
    private Long totalIdleSeconds;
    private Long totalDowntimeSeconds;
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
    
    public static EquipmentDTO fromEntity(Equipment entity) {
        if (entity == null) return null;
        
        EquipmentDTO dto = new EquipmentDTO();
        dto.setId(entity.getId());
        dto.setEquipmentCode(entity.getEquipmentCode());
        dto.setEquipmentName(entity.getEquipmentName());
        dto.setEquipmentTypeId(entity.getEquipmentTypeId());
        dto.setEquipmentTypeCode(entity.getEquipmentTypeCode());
        dto.setWorkshopId(entity.getWorkshopId());
        dto.setWorkstationId(entity.getWorkstationId());
        dto.setStatusCode(entity.getStatusCode());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setStatusConfigId(entity.getStatusConfigId());
        dto.setCurrentTaskNo(entity.getCurrentTaskNo());
        dto.setUtilizationRate(entity.getUtilizationRate());
        dto.setTodayOutput(entity.getTodayOutput());
        dto.setLastMaintenanceTime(entity.getLastMaintenanceTime());
        dto.setNextMaintenanceTime(entity.getNextMaintenanceTime());
        dto.setTotalRunningSeconds(entity.getTotalRunningSeconds());
        dto.setTotalIdleSeconds(entity.getTotalIdleSeconds());
        dto.setTotalDowntimeSeconds(entity.getTotalDowntimeSeconds());
        dto.setDigitalTwinDeviceCode(entity.getDigitalTwinDeviceCode());
        dto.setPositionX(entity.getPositionX());
        dto.setPositionY(entity.getPositionY());
        dto.setPositionZ(entity.getPositionZ());
        dto.setRotationY(entity.getRotationY());
        dto.setIpAddress(entity.getIpAddress());
        dto.setMacAddress(entity.getMacAddress());
        dto.setMqttTopic(entity.getMqttTopic());
        dto.setLastHeartbeat(entity.getLastHeartbeat());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        return dto;
    }
    
    public Equipment toEntity() {
        Equipment equipment = new Equipment();
        if (this.id != null) {
            equipment.setId(this.id);
        }
        equipment.setEquipmentCode(this.equipmentCode);
        equipment.setEquipmentName(this.equipmentName);
        equipment.setEquipmentTypeId(this.equipmentTypeId);
        equipment.setEquipmentTypeCode(this.equipmentTypeCode);
        equipment.setWorkshopId(this.workshopId);
        equipment.setWorkstationId(this.workstationId);
        equipment.setStatusCode(this.statusCode);
        equipment.setStatusConfigId(this.statusConfigId);
        equipment.setCurrentTaskNo(this.currentTaskNo);
        
        return equipment;
    }
    
    // ========== Request DTOs ==========
    
    public static class CreateRequest {
        private String equipmentCode;
        private String equipmentName;
        private Long equipmentTypeId;
        private String equipmentTypeCode;
        private String workshopId;
        private String workstationId;
        
        public String getEquipmentCode() { return equipmentCode; }
        public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
        public String getEquipmentName() { return equipmentName; }
        public void setEquipmentName(String equipmentName) { this.equipmentName = equipmentName; }
        public Long getEquipmentTypeId() { return equipmentTypeId; }
        public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
        public String getEquipmentTypeCode() { return equipmentTypeCode; }
        public void setEquipmentTypeCode(String equipmentTypeCode) { this.equipmentTypeCode = equipmentTypeCode; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getWorkstationId() { return workstationId; }
        public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
    }
    
    public static class UpdateRequest {
        private String equipmentName;
        private Long equipmentTypeId;
        private String equipmentTypeCode;
        private String workshopId;
        private String workstationId;
        private String ipAddress;
        private String macAddress;
        private String mqttTopic;
        
        public String getEquipmentName() { return equipmentName; }
        public void setEquipmentName(String equipmentName) { this.equipmentName = equipmentName; }
        public Long getEquipmentTypeId() { return equipmentTypeId; }
        public void setEquipmentTypeId(Long equipmentTypeId) { this.equipmentTypeId = equipmentTypeId; }
        public String getEquipmentTypeCode() { return equipmentTypeCode; }
        public void setEquipmentTypeCode(String equipmentTypeCode) { this.equipmentTypeCode = equipmentTypeCode; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getWorkstationId() { return workstationId; }
        public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
        public String getIpAddress() { return ipAddress; }
        public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
        public String getMacAddress() { return macAddress; }
        public void setMacAddress(String macAddress) { this.macAddress = macAddress; }
        public String getMqttTopic() { return mqttTopic; }
        public void setMqttTopic(String mqttTopic) { this.mqttTopic = mqttTopic; }
    }
    
    public static class StartTaskRequest {
        private String taskNo;
        
        public String getTaskNo() { return taskNo; }
        public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
    }
    
    public static class BindDigitalTwinRequest {
        private String deviceCode;
        
        public String getDeviceCode() { return deviceCode; }
        public void setDeviceCode(String deviceCode) { this.deviceCode = deviceCode; }
    }
    
    public static class UpdatePositionRequest {
        private Double x;
        private Double y;
        private Double z;
        private Double rotation;
        
        public Double getX() { return x; }
        public void setX(Double x) { this.x = x; }
        public Double getY() { return y; }
        public void setY(Double y) { this.y = y; }
        public Double getZ() { return z; }
        public void setZ(Double z) { this.z = z; }
        public Double getRotation() { return rotation; }
        public void setRotation(Double rotation) { this.rotation = rotation; }
    }
    
    public static class OEECalculationRequest {
        private Integer plannedProductionTime;
        private Integer idealCycleTime;
        private Integer goodProductCount;
        
        public Integer getPlannedProductionTime() { return plannedProductionTime; }
        public void setPlannedProductionTime(Integer plannedProductionTime) { this.plannedProductionTime = plannedProductionTime; }
        public Integer getIdealCycleTime() { return idealCycleTime; }
        public void setIdealCycleTime(Integer idealCycleTime) { this.idealCycleTime = idealCycleTime; }
        public Integer getGoodProductCount() { return goodProductCount; }
        public void setGoodProductCount(Integer goodProductCount) { this.goodProductCount = goodProductCount; }
    }
    
    // ========== Getters and Setters ==========
    
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
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getWorkstationId() { return workstationId; }
    public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
    public String getStatusCode() { return statusCode; }
    public void setStatusCode(String statusCode) { this.statusCode = statusCode; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
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
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}