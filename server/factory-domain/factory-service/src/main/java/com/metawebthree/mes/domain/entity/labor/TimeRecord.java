package com.metawebthree.mes.domain.entity.labor;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

public class TimeRecord {

    public enum RecordType {
        REGULAR, OVERTIME, VACATION, SICK
    }

    public enum RecordStatus {
        DRAFT, SUBMITTED, APPROVED, REJECTED
    }

    private Long id;
    private Long operatorId;
    private String operatorCode;
    private String operatorName;
    private String workOrderNo;
    private String taskNo;
    private String operationCode;
    private String workCenterId;
    private LocalDate recordDate;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private BigDecimal totalHours;
    private RecordType recordType;
    private RecordStatus status;
    private String approvedBy;
    private LocalDateTime approvedAt;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long operatorId, String operatorCode, String operatorName,
                       LocalDate recordDate, LocalDateTime startTime, RecordType recordType) {
        this.operatorId = operatorId;
        this.operatorCode = operatorCode;
        this.operatorName = operatorName;
        this.recordDate = recordDate;
        this.startTime = startTime;
        this.recordType = recordType;
        this.status = RecordStatus.DRAFT;
        this.totalHours = BigDecimal.ZERO;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void clockOut(LocalDateTime endTime) {
        if (endTime.isBefore(this.startTime)) {
            throw new IllegalArgumentException("End time must be after start time");
        }
        this.endTime = endTime;
        long minutes = ChronoUnit.MINUTES.between(this.startTime, endTime);
        this.totalHours = BigDecimal.valueOf(minutes).divide(BigDecimal.valueOf(60), 2, RoundingMode.HALF_UP);
        this.updatedAt = LocalDateTime.now();
    }

    public void assignToTask(String workOrderNo, String taskNo, String operationCode, String workCenterId) {
        this.workOrderNo = workOrderNo;
        this.taskNo = taskNo;
        this.operationCode = operationCode;
        this.workCenterId = workCenterId;
        this.updatedAt = LocalDateTime.now();
    }

    public void submit() {
        if (this.status != RecordStatus.DRAFT) {
            throw new IllegalStateException("Only draft records can be submitted");
        }
        if (this.endTime == null) {
            throw new IllegalStateException("Cannot submit a record without clock-out time");
        }
        this.status = RecordStatus.SUBMITTED;
        this.updatedAt = LocalDateTime.now();
    }

    public void approve(String approvedBy) {
        if (this.status != RecordStatus.SUBMITTED) {
            throw new IllegalStateException("Only submitted records can be approved");
        }
        this.status = RecordStatus.APPROVED;
        this.approvedBy = approvedBy;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void reject(String approvedBy) {
        if (this.status != RecordStatus.SUBMITTED) {
            throw new IllegalStateException("Only submitted records can be rejected");
        }
        this.status = RecordStatus.REJECTED;
        this.approvedBy = approvedBy;
        this.approvedAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getOperatorId() { return operatorId; }
    public void setOperatorId(Long operatorId) { this.operatorId = operatorId; }
    public String getOperatorCode() { return operatorCode; }
    public void setOperatorCode(String operatorCode) { this.operatorCode = operatorCode; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
    public String getWorkOrderNo() { return workOrderNo; }
    public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
    public String getTaskNo() { return taskNo; }
    public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
    public String getOperationCode() { return operationCode; }
    public void setOperationCode(String operationCode) { this.operationCode = operationCode; }
    public String getWorkCenterId() { return workCenterId; }
    public void setWorkCenterId(String workCenterId) { this.workCenterId = workCenterId; }
    public LocalDate getRecordDate() { return recordDate; }
    public void setRecordDate(LocalDate recordDate) { this.recordDate = recordDate; }
    public LocalDateTime getStartTime() { return startTime; }
    public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
    public LocalDateTime getEndTime() { return endTime; }
    public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }
    public BigDecimal getTotalHours() { return totalHours; }
    public void setTotalHours(BigDecimal totalHours) { this.totalHours = totalHours; }
    public RecordType getRecordType() { return recordType; }
    public void setRecordType(RecordType recordType) { this.recordType = recordType; }
    public RecordStatus getStatus() { return status; }
    public void setStatus(RecordStatus status) { this.status = status; }
    public String getApprovedBy() { return approvedBy; }
    public void setApprovedBy(String approvedBy) { this.approvedBy = approvedBy; }
    public LocalDateTime getApprovedAt() { return approvedAt; }
    public void setApprovedAt(LocalDateTime approvedAt) { this.approvedAt = approvedAt; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
