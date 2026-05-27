package com.metawebthree.mes.domain.config;

import lombok.Data;

@Data
public class StatusConfig {
    private Long id;
    private Long machineId;
    private String statusCode;
    private String statusName;
    private String statusCategory;
    private Boolean isInitial;
    private Boolean isFinal;
    private String color;
    private String icon;
    private Integer sortOrder;
}