package com.metawebthree.production.domain.entity;

import java.time.LocalDateTime;

public class ProductionSchedule {
    private Long id;
    private String scheduleCode;
    private String orderCode;
    private String stationCode;
    private Integer sequence;
    private ScheduleStatus status;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private Integer plannedQuantity;
    private Integer completedQuantity;
    private Double progressPercentage;
    private String processRouteCode;
    private Integer processSequence;
    private String requiredSkills;
    private Integer estimatedDuration;
    private Integer actualDuration;
    private String notes;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ScheduleStatus {
        PENDING, READY, IN_PROGRESS, COMPLETED, SKIPPED, DELAYED
    }

    public ProductionSchedule() {
        this.status = ScheduleStatus.PENDING;
        this.progressPercentage = 0.0;
        this.completedQuantity = 0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void startExecution() {
        if (this.status != ScheduleStatus.READY && this.status != ScheduleStatus.PENDING) {
            throw new IllegalStateException("Cannot start schedule in current status");
        }
        this.status = ScheduleStatus.IN_PROGRESS;
        this.actualStartTime = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateProgress(Integer completedQuantity) {
        this.completedQuantity = completedQuantity;
        if (this.plannedQuantity > 0) {
            this.progressPercentage = (double) completedQuantity / this.plannedQuantity * 100;
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void completeSchedule() {
        if (this.status != ScheduleStatus.IN_PROGRESS) {
            throw new IllegalStateException("Can only complete schedules in progress");
        }
        this.status = ScheduleStatus.COMPLETED;
        this.completedQuantity = this.plannedQuantity;
        this.progressPercentage = 100.0;
        this.actualEndTime = LocalDateTime.now();
        
        if (this.actualStartTime != null && this.actualEndTime != null) {
            this.actualDuration = (int) java.time.Duration.between(
                this.actualStartTime, this.actualEndTime
            ).toMinutes();
        }
        
        this.updatedAt = LocalDateTime.now();
    }

    public void skipSchedule() {
        this.status = ScheduleStatus.SKIPPED;
        this.updatedAt = LocalDateTime.now();
    }

    public void markDelayed(String reason) {
        this.status = ScheduleStatus.DELAYED;
        this.notes = (this.notes != null ? this.notes + "; " : "") + "Delayed: " + reason;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isOverdue() {
        if (this.plannedEndTime == null) return false;
        return LocalDateTime.now().isAfter(this.plannedEndTime) 
            && this.status != ScheduleStatus.COMPLETED 
            && this.status != ScheduleStatus.SKIPPED;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getScheduleCode() { return scheduleCode; }
    public void setScheduleCode(String scheduleCode) { this.scheduleCode = scheduleCode; }
    public String getOrderCode() { return orderCode; }
    public void setOrderCode(String orderCode) { this.orderCode = orderCode; }
    public String getStationCode() { return stationCode; }
    public void setStationCode(String stationCode) { this.stationCode = stationCode; }
    public Integer getSequence() { return sequence; }
    public void setSequence(Integer sequence) { this.sequence = sequence; }
    public ScheduleStatus getStatus() { return status; }
    public void setStatus(ScheduleStatus status) { this.status = status; }
    public LocalDateTime getPlannedStartTime() { return plannedStartTime; }
    public void setPlannedStartTime(LocalDateTime plannedStartTime) { this.plannedStartTime = plannedStartTime; }
    public LocalDateTime getPlannedEndTime() { return plannedEndTime; }
    public void setPlannedEndTime(LocalDateTime plannedEndTime) { this.plannedEndTime = plannedEndTime; }
    public LocalDateTime getActualStartTime() { return actualStartTime; }
    public void setActualStartTime(LocalDateTime actualStartTime) { this.actualStartTime = actualStartTime; }
    public LocalDateTime getActualEndTime() { return actualEndTime; }
    public void setActualEndTime(LocalDateTime actualEndTime) { this.actualEndTime = actualEndTime; }
    public Integer getPlannedQuantity() { return plannedQuantity; }
    public void setPlannedQuantity(Integer plannedQuantity) { this.plannedQuantity = plannedQuantity; }
    public Integer getCompletedQuantity() { return completedQuantity; }
    public void setCompletedQuantity(Integer completedQuantity) { this.completedQuantity = completedQuantity; }
    public Double getProgressPercentage() { return progressPercentage; }
    public void setProgressPercentage(Double progressPercentage) { this.progressPercentage = progressPercentage; }
    public String getProcessRouteCode() { return processRouteCode; }
    public void setProcessRouteCode(String processRouteCode) { this.processRouteCode = processRouteCode; }
    public Integer getProcessSequence() { return processSequence; }
    public void setProcessSequence(Integer processSequence) { this.processSequence = processSequence; }
    public String getRequiredSkills() { return requiredSkills; }
    public void setRequiredSkills(String requiredSkills) { this.requiredSkills = requiredSkills; }
    public Integer getEstimatedDuration() { return estimatedDuration; }
    public void setEstimatedDuration(Integer estimatedDuration) { this.estimatedDuration = estimatedDuration; }
    public Integer getActualDuration() { return actualDuration; }
    public void setActualDuration(Integer actualDuration) { this.actualDuration = actualDuration; }
    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}