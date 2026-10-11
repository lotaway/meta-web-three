package com.metawebthree.mes.domain.config;

import lombok.Data;

@Data
public class StatusTransitionRule {
    private Long id;
    private Long machineId;
    private String fromStatus;
    private String toStatus;
    private String transitionAction;
    private String conditionExpression;
    private String eventCode;
    private Boolean isAutoTransition;
    private Integer sortOrder;

    // Explicit getters and setters (Lombok annotation processor not working)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getMachineId() { return machineId; }
    public void setMachineId(Long machineId) { this.machineId = machineId; }
    public String getFromStatus() { return fromStatus; }
    public void setFromStatus(String fromStatus) { this.fromStatus = fromStatus; }
    public String getToStatus() { return toStatus; }
    public void setToStatus(String toStatus) { this.toStatus = toStatus; }
    public String getTransitionAction() { return transitionAction; }
    public void setTransitionAction(String transitionAction) { this.transitionAction = transitionAction; }
    public String getConditionExpression() { return conditionExpression; }
    public void setConditionExpression(String conditionExpression) { this.conditionExpression = conditionExpression; }
    public String getEventCode() { return eventCode; }
    public void setEventCode(String eventCode) { this.eventCode = eventCode; }
    public Boolean getIsAutoTransition() { return isAutoTransition; }
    public void setIsAutoTransition(Boolean isAutoTransition) { this.isAutoTransition = isAutoTransition; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
}