package com.metawebthree.mes.domain.entity.labor;

import java.time.LocalDate;
import java.time.LocalDateTime;

public class WorkCenterAssignment {

    public enum ShiftType {
        DAY, NIGHT, MIDDLE, ROTATING
    }

    public enum AssignmentStatus {
        ACTIVE, INACTIVE
    }

    private Long id;
    private Long operatorId;
    private String workCenterId;
    private String workCenterName;
    private LocalDate startDate;
    private LocalDate endDate;
    private ShiftType shiftType;
    private AssignmentStatus status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long operatorId, String workCenterId, String workCenterName,
                       LocalDate startDate, ShiftType shiftType) {
        this.operatorId = operatorId;
        this.workCenterId = workCenterId;
        this.workCenterName = workCenterName;
        this.startDate = startDate;
        this.shiftType = shiftType;
        this.status = AssignmentStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void assignEndDate(LocalDate endDate) {
        this.endDate = endDate;
        this.updatedAt = LocalDateTime.now();
    }

    public void changeShift(ShiftType newShift) {
        this.shiftType = newShift;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.status = AssignmentStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isActive() {
        return this.status == AssignmentStatus.ACTIVE
            && (endDate == null || !LocalDate.now().isAfter(endDate));
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getOperatorId() { return operatorId; }
    public void setOperatorId(Long operatorId) { this.operatorId = operatorId; }
    public String getWorkCenterId() { return workCenterId; }
    public void setWorkCenterId(String workCenterId) { this.workCenterId = workCenterId; }
    public String getWorkCenterName() { return workCenterName; }
    public void setWorkCenterName(String workCenterName) { this.workCenterName = workCenterName; }
    public LocalDate getStartDate() { return startDate; }
    public void setStartDate(LocalDate startDate) { this.startDate = startDate; }
    public LocalDate getEndDate() { return endDate; }
    public void setEndDate(LocalDate endDate) { this.endDate = endDate; }
    public ShiftType getShiftType() { return shiftType; }
    public void setShiftType(ShiftType shiftType) { this.shiftType = shiftType; }
    public AssignmentStatus getStatus() { return status; }
    public void setStatus(AssignmentStatus status) { this.status = status; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
