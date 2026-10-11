package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class PokaYokeRule {
    private Long id;
    private String ruleCode;
    private String ruleName;
    private RuleType ruleType;
    private RuleStatus status;
    private Long workstationId;
    private String processCode;
    private String productCode;
    private TriggerCondition triggerCondition;
    private List<CheckAction> actions;
    private Integer priority;
    private Boolean enabled;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum RuleType {
        MATERIAL_CHECK,
        SEQUENCE_CHECK,
        PARAMETER_CHECK,
        QUALITY_CHECK
    }

    public enum RuleStatus {
        DRAFT, ACTIVE, INACTIVE
    }

    public enum TriggerCondition {
        ON_MATERIAL_SCAN,
        ON_TASK_START,
        ON_TASK_COMPLETE,
        ON_PARAMETER_RECORD,
        MANUAL_TRIGGER
    }

    public static class CheckAction {
        private ActionType actionType;
        private String actionParam;
        private String message;

        public enum ActionType {
            BLOCK,
            WARNING,
            ANDON_ALERT,
            LOG_ERROR,
            STOP_PROCESS
        }

        public ActionType getActionType() { return actionType; }
        public void setActionType(ActionType actionType) { this.actionType = actionType; }
        public String getActionParam() { return actionParam; }
        public void setActionParam(String actionParam) { this.actionParam = actionParam; }
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    public void create(String ruleCode, String ruleName, RuleType ruleType,
                       Long workstationId, String processCode) {
        this.ruleCode = ruleCode;
        this.ruleName = ruleName;
        this.ruleType = ruleType;
        this.workstationId = workstationId;
        this.processCode = processCode;
        this.status = RuleStatus.DRAFT;
        this.enabled = true;
        this.priority = 0;
        this.actions = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.status = RuleStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void deactivate() {
        this.status = RuleStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void addAction(CheckAction action) {
        if (this.actions == null) {
            this.actions = new ArrayList<>();
        }
        this.actions.add(action);
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isActive() {
        return status == RuleStatus.ACTIVE && enabled != null && enabled;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public RuleType getRuleType() { return ruleType; }
    public void setRuleType(RuleType ruleType) { this.ruleType = ruleType; }
    public RuleStatus getStatus() { return status; }
    public void setStatus(RuleStatus status) { this.status = status; }
    public Long getWorkstationId() { return workstationId; }
    public void setWorkstationId(Long workstationId) { this.workstationId = workstationId; }
    public String getProcessCode() { return processCode; }
    public void setProcessCode(String processCode) { this.processCode = processCode; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public TriggerCondition getTriggerCondition() { return triggerCondition; }
    public void setTriggerCondition(TriggerCondition triggerCondition) { this.triggerCondition = triggerCondition; }
    public List<CheckAction> getActions() { return actions; }
    public void setActions(List<CheckAction> actions) { this.actions = actions; }
    public Integer getPriority() { return priority; }
    public void setPriority(Integer priority) { this.priority = priority; }
    public Boolean getEnabled() { return enabled; }
    public void setEnabled(Boolean enabled) { this.enabled = enabled; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}