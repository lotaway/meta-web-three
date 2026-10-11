package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_flow_instance")
public class ProcessFlowInstanceDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String instanceCode;
    private Long templateId;
    private String templateName;
    private String businessType;
    private String businessKey;
    private String currentNodeId;
    private String currentNodeName;
    private String status;
    private String flowData;
    private LocalDateTime startedAt;
    private Long startedBy;
    private LocalDateTime completedAt;
    private Long completedBy;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;

    // Explicit getters and setters (Lombok annotation processor not working)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getInstanceCode() { return instanceCode; }
    public void setInstanceCode(String instanceCode) { this.instanceCode = instanceCode; }
    public Long getTemplateId() { return templateId; }
    public void setTemplateId(Long templateId) { this.templateId = templateId; }
    public String getTemplateName() { return templateName; }
    public void setTemplateName(String templateName) { this.templateName = templateName; }
    public String getBusinessType() { return businessType; }
    public void setBusinessType(String businessType) { this.businessType = businessType; }
    public String getBusinessKey() { return businessKey; }
    public void setBusinessKey(String businessKey) { this.businessKey = businessKey; }
    public String getCurrentNodeId() { return currentNodeId; }
    public void setCurrentNodeId(String currentNodeId) { this.currentNodeId = currentNodeId; }
    public String getCurrentNodeName() { return currentNodeName; }
    public void setCurrentNodeName(String currentNodeName) { this.currentNodeName = currentNodeName; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getFlowData() { return flowData; }
    public void setFlowData(String flowData) { this.flowData = flowData; }
    public LocalDateTime getStartedAt() { return startedAt; }
    public void setStartedAt(LocalDateTime startedAt) { this.startedAt = startedAt; }
    public Long getStartedBy() { return startedBy; }
    public void setStartedBy(Long startedBy) { this.startedBy = startedBy; }
    public LocalDateTime getCompletedAt() { return completedAt; }
    public void setCompletedAt(LocalDateTime completedAt) { this.completedAt = completedAt; }
    public Long getCompletedBy() { return completedBy; }
    public void setCompletedBy(Long completedBy) { this.completedBy = completedBy; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public Long getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(Long updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public Boolean getDeleted() { return deleted; }
    public void setDeleted(Boolean deleted) { this.deleted = deleted; }
}