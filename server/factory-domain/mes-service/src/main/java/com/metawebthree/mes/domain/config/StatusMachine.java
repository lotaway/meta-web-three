package com.metawebthree.mes.domain.config;

import lombok.Data;
import java.util.List;

@Data
public class StatusMachine {
    private Long id;
    private String machineCode;
    private String machineName;
    private String entityType;
    private String description;
    private String initialStatus;
    private Boolean isDefault;
    private String status;
    private List<StatusConfig> statuses;
    private List<StatusTransitionRule> transitions;

    // Explicit getters and setters (Lombok annotation processor not working)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getMachineCode() { return machineCode; }
    public void setMachineCode(String machineCode) { this.machineCode = machineCode; }
    public String getMachineName() { return machineName; }
    public void setMachineName(String machineName) { this.machineName = machineName; }
    public String getEntityType() { return entityType; }
    public void setEntityType(String entityType) { this.entityType = entityType; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getInitialStatus() { return initialStatus; }
    public void setInitialStatus(String initialStatus) { this.initialStatus = initialStatus; }
    public Boolean getIsDefault() { return isDefault; }
    public void setIsDefault(Boolean isDefault) { this.isDefault = isDefault; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public List<StatusConfig> getStatuses() { return statuses; }
    public void setStatuses(List<StatusConfig> statuses) { this.statuses = statuses; }
    public List<StatusTransitionRule> getTransitions() { return transitions; }
    public void setTransitions(List<StatusTransitionRule> transitions) { this.transitions = transitions; }
}