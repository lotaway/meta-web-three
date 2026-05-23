package com.metawebthree.production.domain.entity;

import java.time.LocalDateTime;

public class WorkStation {
    private Long id;
    private String stationCode;
    private String stationName;
    private String stationType;
    private String workshopCode;
    private String productionLineCode;
    private StationStatus status;
    private Integer capacity;
    private Integer currentLoad;
    private Double efficiency;
    private String currentOperator;
    private String currentOrderCode;
    private Double positionX;
    private Double positionY;
    private String ipAddress;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    private String equipmentList;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum StationStatus {
        IDLE, OPERATING, MAINTENANCE, BREAKDOWN, OFFLINE
    }

    public WorkStation() {
        this.status = StationStatus.IDLE;
        this.currentLoad = 0;
        this.efficiency = 0.0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public boolean canAcceptOrder() {
        return this.status == StationStatus.IDLE 
            && this.currentLoad < this.capacity;
    }

    public void assignOrder(String orderCode) {
        if (!canAcceptOrder()) {
            throw new IllegalStateException("Work station cannot accept new order");
        }
        this.currentOrderCode = orderCode;
        this.status = StationStatus.OPERATING;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeOrder() {
        this.currentOrderCode = null;
        this.status = StationStatus.IDLE;
        this.updatedAt = LocalDateTime.now();
    }

    public void startMaintenance() {
        this.status = StationStatus.MAINTENANCE;
        this.currentOrderCode = null;
        this.updatedAt = LocalDateTime.now();
    }

    public void completeMaintenance() {
        this.status = StationStatus.IDLE;
        this.lastMaintenanceTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reportBreakdown(String reason) {
        this.status = StationStatus.BREAKDOWN;
        this.updatedAt = LocalDateTime.now();
    }

    public void repair() {
        this.status = StationStatus.IDLE;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateEfficiency(Double newEfficiency) {
        this.efficiency = newEfficiency;
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getStationCode() { return stationCode; }
    public void setStationCode(String stationCode) { this.stationCode = stationCode; }
    public String getStationName() { return stationName; }
    public void setStationName(String stationName) { this.stationName = stationName; }
    public String getStationType() { return stationType; }
    public void setStationType(String stationType) { this.stationType = stationType; }
    public String getWorkshopCode() { return workshopCode; }
    public void setWorkshopCode(String workshopCode) { this.workshopCode = workshopCode; }
    public String getProductionLineCode() { return productionLineCode; }
    public void setProductionLineCode(String productionLineCode) { this.productionLineCode = productionLineCode; }
    public StationStatus getStatus() { return status; }
    public void setStatus(StationStatus status) { this.status = status; }
    public Integer getCapacity() { return capacity; }
    public void setCapacity(Integer capacity) { this.capacity = capacity; }
    public Integer getCurrentLoad() { return currentLoad; }
    public void setCurrentLoad(Integer currentLoad) { this.currentLoad = currentLoad; }
    public Double getEfficiency() { return efficiency; }
    public void setEfficiency(Double efficiency) { this.efficiency = efficiency; }
    public String getCurrentOperator() { return currentOperator; }
    public void setCurrentOperator(String currentOperator) { this.currentOperator = currentOperator; }
    public String getCurrentOrderCode() { return currentOrderCode; }
    public void setCurrentOrderCode(String currentOrderCode) { this.currentOrderCode = currentOrderCode; }
    public Double getPositionX() { return positionX; }
    public void setPositionX(Double positionX) { this.positionX = positionX; }
    public Double getPositionY() { return positionY; }
    public void setPositionY(Double positionY) { this.positionY = positionY; }
    public String getIpAddress() { return ipAddress; }
    public void setIpAddress(String ipAddress) { this.ipAddress = ipAddress; }
    public LocalDateTime getLastMaintenanceTime() { return lastMaintenanceTime; }
    public void setLastMaintenanceTime(LocalDateTime lastMaintenanceTime) { this.lastMaintenanceTime = lastMaintenanceTime; }
    public LocalDateTime getNextMaintenanceTime() { return nextMaintenanceTime; }
    public void setNextMaintenanceTime(LocalDateTime nextMaintenanceTime) { this.nextMaintenanceTime = nextMaintenanceTime; }
    public String getEquipmentList() { return equipmentList; }
    public void setEquipmentList(String equipmentList) { this.equipmentList = equipmentList; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}