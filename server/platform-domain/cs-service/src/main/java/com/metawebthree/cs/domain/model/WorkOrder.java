package com.metawebthree.cs.domain.model;

import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import com.metawebthree.cs.domain.model.enums.WorkOrderPriority;
import com.metawebthree.cs.domain.model.enums.WorkOrderStatus;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
public class WorkOrder {
    private Long id;
    private String orderNo;
    private Long customerId;
    private Long agentId;
    private WorkOrderCategory category;
    private WorkOrderCategory aiSuggestedCategory;
    private Double confidenceScore;
    private WorkOrderStatus status;
    private WorkOrderPriority priority;
    private String title;
    private String description;
    private String resolution;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
    private LocalDateTime resolveTime;
    private Map<String, Object> metadata;

    public WorkOrder() {}

    public WorkOrder(Long customerId, String title, String description, WorkOrderCategory category) {
        this.customerId = customerId;
        this.title = title;
        this.description = description;
        this.category = category;
        this.status = WorkOrderStatus.PENDING;
        this.priority = WorkOrderPriority.MEDIUM;
        this.confidenceScore = 0.0;
        this.createTime = LocalDateTime.now();
        this.updateTime = LocalDateTime.now();
    }

    public boolean needsEscalation() {
        return this.priority == WorkOrderPriority.URGENT 
            || this.status == WorkOrderStatus.ESCALATED;
    }

    public void escalate() {
        this.status = WorkOrderStatus.ESCALATED;
        this.priority = WorkOrderPriority.URGENT;
        this.updateTime = LocalDateTime.now();
    }

    public void resolve(String resolution) {
        this.status = WorkOrderStatus.RESOLVED;
        this.resolution = resolution;
        this.resolveTime = LocalDateTime.now();
        this.updateTime = LocalDateTime.now();
    }

    public void assign(Long agentId) {
        this.agentId = agentId;
        this.status = WorkOrderStatus.PROCESSING;
        this.updateTime = LocalDateTime.now();
    }

    public void updateCategory(WorkOrderCategory category, Double confidence) {
        this.aiSuggestedCategory = category;
        this.confidenceScore = confidence;
        this.updateTime = LocalDateTime.now();
    }
}