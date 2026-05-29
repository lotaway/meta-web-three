package com.metawebthree.mes.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class WorkReport {
    
    private Long id;
    private String reportNo;
    private Long taskId;
    private String taskNo;
    private Long workOrderId;
    private String workOrderNo;
    private Long workstationId;
    private String workstationName;
    private String processCode;
    private String processName;
    private Integer stepNo;
    private String operatorId;
    private String operatorName;
    private LocalDateTime reportTime;
    private Integer quantity;
    private Integer qualifiedQuantity;
    private Integer defectiveQuantity;
    private Integer durationMinutes;
    private String parameterValuesJson;
    private String remarks;
    private ReportStatus status;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public enum ReportStatus {
        DRAFT, SUBMITTED, QUALITY_CHECKED, CONFIRMED, CANCELLED
    }
    
    public void create(String reportNo, Long taskId, String taskNo, Long workOrderId,
                      String workOrderNo, Long workstationId, String workstationName,
                      String processCode, String processName, Integer stepNo,
                      String operatorId, String operatorName) {
        this.reportNo = reportNo;
        this.taskId = taskId;
        this.taskNo = taskNo;
        this.workOrderId = workOrderId;
        this.workOrderNo = workOrderNo;
        this.workstationId = workstationId;
        this.workstationName = workstationName;
        this.processCode = processCode;
        this.processName = processName;
        this.stepNo = stepNo;
        this.operatorId = operatorId;
        this.operatorName = operatorName;
        this.reportTime = LocalDateTime.now();
        this.qualifiedQuantity = 0;
        this.defectiveQuantity = 0;
        this.quantity = 0;
        this.durationMinutes = 0;
        this.status = ReportStatus.DRAFT;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void submit() {
        if (status != ReportStatus.DRAFT) {
            throw new IllegalStateException("Only DRAFT status can be submitted");
        }
        if (quantity == null || quantity <= 0) {
            throw new IllegalArgumentException("Quantity must be greater than 0");
        }
        this.status = ReportStatus.SUBMITTED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void qualityChecked() {
        if (status != ReportStatus.SUBMITTED) {
            throw new IllegalStateException("Only SUBMITTED status can be quality checked");
        }
        this.status = ReportStatus.QUALITY_CHECKED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void confirm() {
        if (status != ReportStatus.QUALITY_CHECKED) {
            throw new IllegalStateException("Only QUALITY_CHECKED status can be confirmed");
        }
        this.status = ReportStatus.CONFIRMED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void cancel() {
        if (status == ReportStatus.CONFIRMED) {
            throw new IllegalStateException("CONFIRMED report cannot be cancelled");
        }
        this.status = ReportStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void recordOutput(Integer quantity, Integer qualified, Integer defective, Integer duration) {
        this.quantity = quantity;
        this.qualifiedQuantity = qualified;
        this.defectiveQuantity = defective;
        this.durationMinutes = duration;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void setParameterValues(String json) {
        this.parameterValuesJson = json;
        this.updatedAt = LocalDateTime.now();
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getReportNo() { return reportNo; }
    public void setReportNo(String reportNo) { this.reportNo = reportNo; }
    public Long getTaskId() { return taskId; }
    public void setTaskId(Long taskId) { this.taskId = taskId; }
    public String getTaskNo() { return taskNo; }
    public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
    public Long getWorkOrderId() { return workOrderId; }
    public void setWorkOrderId(Long workOrderId) { this.workOrderId = workOrderId; }
    public String getWorkOrderNo() { return workOrderNo; }
    public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
    public Long getWorkstationId() { return workstationId; }
    public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
    public String getWorkstationName() { return workstationName; }
    public void setWorkstationName(String workstationName) { this.workstationName = workstationName; }
    public String getProcessCode() { return processCode; }
    public void setProcessCode(String processCode) { this.processCode = processCode; }
    public String getProcessName() { return processName; }
    public void setProcessName(String processName) { this.processName = processName; }
    public Integer getStepNo() { return stepNo; }
    public void setStepNo(Integer stepNo) { this.stepNo = stepNo; }
    public String getOperatorId() { return operatorId; }
    public void setOperatorId(String operatorId) { this.operatorId = operatorId; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
    public LocalDateTime getReportTime() { return reportTime; }
    public void setReportTime(LocalDateTime reportTime) { this.reportTime = reportTime; }
    public Integer getQuantity() { return quantity; }
    public void setQuantity(Integer quantity) { this.quantity = quantity; }
    public Integer getQualifiedQuantity() { return qualifiedQuantity; }
    public void setQualifiedQuantity(Integer qualifiedQuantity) { this.qualifiedQuantity = qualifiedQuantity; }
    public Integer getDefectiveQuantity() { return defectiveQuantity; }
    public void setDefectiveQuantity(Integer defectiveQuantity) { this.defectiveQuantity = defectiveQuantity; }
    public Integer getDurationMinutes() { return durationMinutes; }
    public void setDurationMinutes(Integer durationMinutes) { this.durationMinutes = durationMinutes; }
    public String getParameterValuesJson() { return parameterValuesJson; }
    public void setParameterValuesJson(String parameterValuesJson) { this.parameterValuesJson = parameterValuesJson; }
    public String getRemarks() { return remarks; }
    public void setRemarks(String remarks) { this.remarks = remarks; }
    public ReportStatus getStatus() { return status; }
    public void setStatus(ReportStatus status) { this.status = status; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}