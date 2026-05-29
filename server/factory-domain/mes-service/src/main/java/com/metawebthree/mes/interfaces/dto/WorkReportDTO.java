package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.WorkReport;

import java.time.LocalDateTime;

public class WorkReportDTO {
    
    private Long id;
    private String reportNo;
    private Long taskId;
    private String taskNo;
    private Long workOrderId;
    private String workOrderNo;
    private String workstationId;
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
    private String status;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static WorkReportDTO fromEntity(WorkReport report) {
        WorkReportDTO dto = new WorkReportDTO();
        dto.setId(report.getId());
        dto.setReportNo(report.getReportNo());
        dto.setTaskId(report.getTaskId());
        dto.setTaskNo(report.getTaskNo());
        dto.setWorkOrderId(report.getWorkOrderId());
        dto.setWorkOrderNo(report.getWorkOrderNo());
        dto.setWorkstationId(report.getWorkstationId());
        dto.setWorkstationName(report.getWorkstationName());
        dto.setProcessCode(report.getProcessCode());
        dto.setProcessName(report.getProcessName());
        dto.setStepNo(report.getStepNo());
        dto.setOperatorId(report.getOperatorId());
        dto.setOperatorName(report.getOperatorName());
        dto.setReportTime(report.getReportTime());
        dto.setQuantity(report.getQuantity());
        dto.setQualifiedQuantity(report.getQualifiedQuantity());
        dto.setDefectiveQuantity(report.getDefectiveQuantity());
        dto.setDurationMinutes(report.getDurationMinutes());
        dto.setParameterValuesJson(report.getParameterValuesJson());
        dto.setRemarks(report.getRemarks());
        dto.setStatus(report.getStatus() != null ? report.getStatus().name() : null);
        dto.setCreatedBy(report.getCreatedBy());
        dto.setCreatedAt(report.getCreatedAt());
        dto.setUpdatedAt(report.getUpdatedAt());
        return dto;
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
    public String getWorkstationId() { return workstationId; }
    public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
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
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public static class CreateRequest {
        private Long taskId;
        private String taskNo;
        private Long workOrderId;
        private String workOrderNo;
        private String workstationId;
        private String workstationName;
        private String processCode;
        private String processName;
        private Integer stepNo;
        private String operatorId;
        private String operatorName;
        private Integer quantity;
        private Integer qualifiedQuantity;
        private Integer defectiveQuantity;
        private Integer durationMinutes;
        private String parameterValuesJson;
        private String remarks;
        
        public Long getTaskId() { return taskId; }
        public void setTaskId(Long taskId) { this.taskId = taskId; }
        public String getTaskNo() { return taskNo; }
        public void setTaskNo(String taskNo) { this.taskNo = taskNo; }
        public Long getWorkOrderId() { return workOrderId; }
        public void setWorkOrderId(Long workOrderId) { this.workOrderId = workOrderId; }
        public String getWorkOrderNo() { return workOrderNo; }
        public void setWorkOrderNo(String workOrderNo) { this.workOrderNo = workOrderNo; }
        public String getWorkstationId() { return workstationId; }
        public void setWorkstationId(String workstationId) { this.workstationId = workstationId; }
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
    }
}