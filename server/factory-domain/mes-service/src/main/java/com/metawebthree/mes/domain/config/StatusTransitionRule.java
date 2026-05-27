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
}