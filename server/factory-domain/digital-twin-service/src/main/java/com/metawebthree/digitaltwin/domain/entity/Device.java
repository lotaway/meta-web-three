package com.metawebthree.digitaltwin.domain.entity;

import java.time.LocalDateTime;

public class Device {
    private Long id;
    private String deviceCode;
    private String deviceName;
    private String deviceType;
    private String workshopId;
    private String productionLineId;
    private DeviceStatus status;
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

    public enum DeviceStatus {
        ONLINE, OFFLINE, RUNNING, IDLE, WARNING, ERROR, MAINTENANCE
    }

    public void create(String deviceCode, String deviceName, String deviceType, 
                      String workshopId, String productionLineId) {
        this.deviceCode = deviceCode;
        this.deviceName = deviceName;
        this.deviceType = deviceType;
        this.workshopId = workshopId;
        this.productionLineId = productionLineId;
        this.status = DeviceStatus.OFFLINE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void goOnline() {
        this.status = DeviceStatus.ONLINE;
        this.lastHeartbeat = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void goOffline() {
        this.status = DeviceStatus.OFFLINE;
        this.updatedAt = LocalDateTime.now();
    }

    public void startRunning() {
        this.status = DeviceStatus.RUNNING;
        this.lastHeartbeat = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void setIdle() {
        this.status = DeviceStatus.IDLE;
        this.lastHeartbeat = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reportError() {
        this.status = DeviceStatus.ERROR;
        this.updatedAt = LocalDateTime.now();
    }

    public void setWarning() {
        this.status = DeviceStatus.WARNING;
        this.updatedAt = LocalDateTime.now();
    }

    public void enterMaintenance() {
        this.status = DeviceStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public void updatePosition(Double x, Double y, Double z, Double rotation) {
        this.positionX = x;
        this.positionY = y;
        this.positionZ = z;
        this.rotationY = rotation;
        this.updatedAt = LocalDateTime.now();
    }

    public void heartbeat() {
        this.lastHeartbeat = LocalDateTime.now();
        if (this.status == DeviceStatus.OFFLINE) {
            this.status = DeviceStatus.ONLINE;
        }
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDeviceCode() { return deviceCode; }
    public void setDeviceCode(String deviceCode) { this.deviceCode = deviceCode; }
    public String getDeviceName() { return deviceName; }
    public void setDeviceName(String deviceName) { this.deviceName = deviceName; }
    public String getDeviceType() { return deviceType; }
    public void setDeviceType(String deviceType) { this.deviceType = deviceType; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getProductionLineId() { return productionLineId; }
    public void setProductionLineId(String productionLineId) { this.productionLineId = productionLineId; }
    public DeviceStatus getStatus() { return status; }
    public void setStatus(DeviceStatus status) { this.status = status; }
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

    // equals/hashCode based on business key (deviceCode)
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Device device = (Device) o;
        return deviceCode != null && deviceCode.equals(device.deviceCode);
    }

    @Override
    public int hashCode() {
        return deviceCode != null ? deviceCode.hashCode() : 0;
    }
}