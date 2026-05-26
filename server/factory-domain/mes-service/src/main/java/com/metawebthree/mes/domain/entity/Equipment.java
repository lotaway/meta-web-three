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
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum EquipmentStatus {
        IDLE, RUNNING, MAINTENANCE, BREAKDOWN, SCRAP
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
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}