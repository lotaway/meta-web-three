package com.metawebthree.digitaltwin.domain.entity;

import java.time.LocalDateTime;

public class Workshop {
    private Long id;
    private String workshopCode;
    private String workshopName;
    private String description;
    private WorkshopStatus status;
    private Integer area;
    private String location;
    private Double centerX;
    private Double centerY;
    private Double width;
    private Double length;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum WorkshopStatus {
        PLANNING, CONSTRUCTION, OPERATING, MAINTENANCE, DECOMMISSIONED
    }

    public void create(String workshopCode, String workshopName, String description) {
        this.workshopCode = workshopCode;
        this.workshopName = workshopName;
        this.description = description;
        this.status = WorkshopStatus.PLANNING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void startConstruction() {
        this.status = WorkshopStatus.CONSTRUCTION;
        this.updatedAt = LocalDateTime.now();
    }

    public void startOperating() {
        this.status = WorkshopStatus.OPERATING;
        this.updatedAt = LocalDateTime.now();
    }

    public void enterMaintenance() {
        this.status = WorkshopStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public void decommission() {
        this.status = WorkshopStatus.DECOMMISSIONED;
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getWorkshopCode() { return workshopCode; }
    public void setWorkshopCode(String workshopCode) { this.workshopCode = workshopCode; }
    public String getWorkshopName() { return workshopName; }
    public void setWorkshopName(String workshopName) { this.workshopName = workshopName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public WorkshopStatus getStatus() { return status; }
    public void setStatus(WorkshopStatus status) { this.status = status; }
    public Integer getArea() { return area; }
    public void setArea(Integer area) { this.area = area; }
    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }
    public Double getCenterX() { return centerX; }
    public void setCenterX(Double centerX) { this.centerX = centerX; }
    public Double getCenterY() { return centerY; }
    public void setCenterY(Double centerY) { this.centerY = centerY; }
    public Double getWidth() { return width; }
    public void setWidth(Double width) { this.width = width; }
    public Double getLength() { return length; }
    public void setLength(Double length) { this.length = length; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // equals/hashCode based on business key (workshopCode)
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Workshop workshop = (Workshop) o;
        return workshopCode != null && workshopCode.equals(workshop.workshopCode);
    }

    @Override
    public int hashCode() {
        return workshopCode != null ? workshopCode.hashCode() : 0;
    }
}