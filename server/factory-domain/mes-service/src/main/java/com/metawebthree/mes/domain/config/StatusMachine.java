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
}