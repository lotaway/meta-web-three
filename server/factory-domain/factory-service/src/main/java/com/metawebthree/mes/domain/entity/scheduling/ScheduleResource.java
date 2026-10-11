package com.metawebthree.mes.domain.entity.scheduling;

import java.time.LocalDateTime;
import java.util.List;

public class ScheduleResource {

    public enum ResourceType {
        EQUIPMENT, WORK_CENTER, LABOR, TOOL
    }

    public enum ResourceStatus {
        AVAILABLE, OCCUPIED, MAINTENANCE, OFFLINE
    }

    private Long id;
    private String resourceCode;
    private String resourceName;
    private ResourceType resourceType;
    private ResourceStatus status;
    private String workshopId;
    private Double capacityPerShift;
    private String calendarCode;
    private List<TimeSlot> occupiedSlots;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static class TimeSlot {
        private LocalDateTime startTime;
        private LocalDateTime endTime;
        private Long scheduleOrderId;
        private String scheduleNo;

        public TimeSlot(LocalDateTime startTime, LocalDateTime endTime, Long scheduleOrderId, String scheduleNo) {
            this.startTime = startTime;
            this.endTime = endTime;
            this.scheduleOrderId = scheduleOrderId;
            this.scheduleNo = scheduleNo;
        }

        public LocalDateTime getStartTime() { return startTime; }
        public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
        public LocalDateTime getEndTime() { return endTime; }
        public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
        public Long getScheduleOrderId() { return scheduleOrderId; }
        public void setScheduleOrderId(Long scheduleOrderId) { this.scheduleOrderId = scheduleOrderId; }
        public String getScheduleNo() { return scheduleNo; }
        public void setScheduleNo(String scheduleNo) { this.scheduleNo = scheduleNo; }
    }

    public void create(String resourceCode, String resourceName, ResourceType resourceType, String workshopId) {
        this.resourceCode = resourceCode;
        this.resourceName = resourceName;
        this.resourceType = resourceType;
        this.workshopId = workshopId;
        this.status = ResourceStatus.AVAILABLE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isAvailable(LocalDateTime start, LocalDateTime end) {
        if (status != ResourceStatus.AVAILABLE) return false;
        if (occupiedSlots == null) return true;
        for (TimeSlot slot : occupiedSlots) {
            if (start.isBefore(slot.getEndTime()) && end.isAfter(slot.getStartTime())) {
                return false;
            }
        }
        return true;
    }

    public void occupy(LocalDateTime start, LocalDateTime end, Long orderId, String scheduleNo) {
        if (occupiedSlots == null) {
            occupiedSlots = new java.util.ArrayList<>();
        }
        occupiedSlots.add(new TimeSlot(start, end, orderId, scheduleNo));
        this.status = ResourceStatus.OCCUPIED;
        this.updatedAt = LocalDateTime.now();
    }

    public void release(LocalDateTime endTime) {
        if (occupiedSlots != null) {
            occupiedSlots.removeIf(slot -> slot.getEndTime().isBefore(endTime) || slot.getEndTime().equals(endTime));
            if (occupiedSlots.isEmpty()) {
                this.status = ResourceStatus.AVAILABLE;
            }
        }
        this.updatedAt = LocalDateTime.now();
    }

    public void setMaintenance() {
        this.status = ResourceStatus.MAINTENANCE;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getResourceCode() { return resourceCode; }
    public void setResourceCode(String resourceCode) { this.resourceCode = resourceCode; }
    public String getResourceName() { return resourceName; }
    public void setResourceName(String resourceName) { this.resourceName = resourceName; }
    public ResourceType getResourceType() { return resourceType; }
    public void setResourceType(ResourceType resourceType) { this.resourceType = resourceType; }
    public ResourceStatus getStatus() { return status; }
    public void setStatus(ResourceStatus status) { this.status = status; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public Double getCapacityPerShift() { return capacityPerShift; }
    public void setCapacityPerShift(Double capacityPerShift) { this.capacityPerShift = capacityPerShift; }
    public String getCalendarCode() { return calendarCode; }
    public void setCalendarCode(String calendarCode) { this.calendarCode = calendarCode; }
    public List<TimeSlot> getOccupiedSlots() { return occupiedSlots; }
    public void setOccupiedSlots(List<TimeSlot> occupiedSlots) { this.occupiedSlots = occupiedSlots; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
