package com.metawebthree.mes.domain.config;

import lombok.Data;

@Data
public class WorkOrderType {
    private Long id;
    private String typeCode;
    private String typeName;
    private String description;
    private String statusMachineCode;
    private String processRouteTemplate;
    private Boolean isDefault;
    private Integer sortOrder;
    private String status;
    
    public static final String TYPE_NORMAL = "NORMAL";
    public static final String TYPE_REWORK = "REWORK";
    public static final String TYPE_REPAIR = "REPAIR";
    public static final String TYPE_SAMPLE = "SAMPLE";
}