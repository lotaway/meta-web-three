package com.metawebthree.digitaltwin.domain.entity;

import java.time.LocalDateTime;

public class ProductionLine {
    private Long id;
    private String lineCode;
    private String lineName;
    private String workshopId;
    private ProductionLineStatus status;
    private Integer capacity;
    private Integer currentOutput;
    private Double efficiency;
    private String productTypes;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ProductionLineStatus {
        IDLE, RUNNING, PAUSED, MAINTENANCE, BROKEN_DOWN
    }

    public void create(String lineCode, String lineName, String workshopId, Integer capacity) {
        this.lineCode = lineCode;
        this.lineName = lineName;
        this.workshopId = workshopId;
        this.capacity = capacity;
        this.status = ProductionLineStatus.IDLE;
        this.currentOutput = 0;
        this.efficiency = 0.0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void start() {
        this.status = ProductionLineStatus.RUNNING;
        this.updatedAt = LocalDateTime.now();
    }

    public void pause() {
        this.status = ProductionLineStatus.PAUSED;
        this.updatedAt = LocalDateTime.now();
    }

    public void resume() {
        this.status = ProductionLineStatus.RUNNING;
        this.updatedAt = LocalDateTime.now();
    }

    public void stop() {
        this.status = ProductionLineStatus.IDLE;
        this.updatedAt = LocalDateTime.now();
    }

    public void maintenance() {
        this.status = ProductionLineStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public void breakdown() {
        this.status = ProductionLineStatus.BROKEN_DOWN;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateOutput(Integer output) {
        this.currentOutput = output;
        if (capacity > 0) {
            this.efficiency = (double) output / capacity * 100;
        }
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getLineCode() { return lineCode; }
    public void setLineCode(String lineCode) { this.lineCode = lineCode; }
    public String getLineName() { return lineName; }
    public void setLineName(String lineName) { this.lineName = lineName; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public ProductionLineStatus getStatus() { return status; }
    public void setStatus(ProductionLineStatus status) { this.status = status; }
    public Integer getCapacity() { return capacity; }
    public void setCapacity(Integer capacity) { this.capacity = capacity; }
    public Integer getCurrentOutput() { return currentOutput; }
    public void setCurrentOutput(Integer currentOutput) { this.currentOutput = currentOutput; }
    public Double getEfficiency() { return efficiency; }
    public void setEfficiency(Double efficiency) { this.efficiency = efficiency; }
    public String getProductTypes() { return productTypes; }
    public void setProductTypes(String productTypes) { this.productTypes = productTypes; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}