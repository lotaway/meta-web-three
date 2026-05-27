package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.List;

public class WorkOrderSplitRule {
    private Long id;
    private String ruleName;
    private String ruleCode;
    private String splitType;
    private String description;
    private Boolean enabled;
    private Integer splitQuantity;
    private Integer maxChildOrders;
    private String bomId;
    private String processRouteId;
    private List<SplitCondition> conditions;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static class SplitCondition {
        private String field;
        private String operator;
        private Object value;
        private String logic;

        public String getField() { return field; }
        public void setField(String field) { this.field = field; }
        public String getOperator() { return operator; }
        public void setOperator(String operator) { this.operator = operator; }
        public Object getValue() { return value; }
        public void setValue(Object value) { this.value = value; }
        public String getLogic() { return logic; }
        public void setLogic(String logic) { this.logic = logic; }
    }

    public void create(String ruleName, String ruleCode, String splitType) {
        this.ruleName = ruleName;
        this.ruleCode = ruleCode;
        this.splitType = splitType;
        this.enabled = true;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateConditions(List<SplitCondition> conditions) {
        this.conditions = conditions;
        this.updatedAt = LocalDateTime.now();
    }

    public void enable() {
        this.enabled = true;
        this.updatedAt = LocalDateTime.now();
    }

    public void disable() {
        this.enabled = false;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isValid() {
        return ruleName != null && ruleCode != null && splitType != null && enabled;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRuleName() { return ruleName; }
    public void setRuleName(String ruleName) { this.ruleName = ruleName; }
    public String getRuleCode() { return ruleCode; }
    public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
    public String getSplitType() { return splitType; }
    public void setSplitType(String splitType) { this.splitType = splitType; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Boolean getEnabled() { return enabled; }
    public void setEnabled(Boolean enabled) { this.enabled = enabled; }
    public Integer getSplitQuantity() { return splitQuantity; }
    public void setSplitQuantity(Integer splitQuantity) { this.splitQuantity = splitQuantity; }
    public Integer getMaxChildOrders() { return maxChildOrders; }
    public void setMaxChildOrders(Integer maxChildOrders) { this.maxChildOrders = maxChildOrders; }
    public String getBomId() { return bomId; }
    public void setBomId(String bomId) { this.bomId = bomId; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public List<SplitCondition> getConditions() { return conditions; }
    public void setConditions(List<SplitCondition> conditions) { this.conditions = conditions; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}