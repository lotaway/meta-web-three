package com.metawebthree.mes.common;

public final class MesPermissions {

    private MesPermissions() {}

    // Work Order permissions
    public static final String WORK_ORDER_READ   = "mes:work-order:read";
    public static final String WORK_ORDER_CREATE = "mes:work-order:create";
    public static final String WORK_ORDER_UPDATE = "mes:work-order:update";
    public static final String WORK_ORDER_DELETE = "mes:work-order:delete";
    public static final String WORK_ORDER_RELEASE = "mes:work-order:release";

    // Task permissions
    public static final String TASK_READ   = "mes:task:read";
    public static final String TASK_CREATE = "mes:task:create";
    public static final String TASK_UPDATE = "mes:task:update";
    public static final String TASK_START  = "mes:task:start";
    public static final String TASK_COMPLETE = "mes:task:complete";

    // Equipment permissions
    public static final String EQUIPMENT_READ   = "mes:equipment:read";
    public static final String EQUIPMENT_CREATE = "mes:equipment:create";
    public static final String EQUIPMENT_UPDATE = "mes:equipment:update";
    public static final String EQUIPMENT_DELETE = "mes:equipment:delete";
    public static final String EQUIPMENT_BREAKDOWN = "mes:equipment:breakdown";
    public static final String EQUIPMENT_REPAIR  = "mes:equipment:repair";

    // Process Route permissions
    public static final String PROCESS_ROUTE_READ   = "mes:process-route:read";
    public static final String PROCESS_ROUTE_CREATE = "mes:process-route:create";
    public static final String PROCESS_ROUTE_UPDATE = "mes:process-route:update";
    public static final String PROCESS_ROUTE_DELETE = "mes:process-route:delete";

    // Process Parameter permissions
    public static final String PROCESS_PARAMETER_READ   = "mes:process-parameter:read";
    public static final String PROCESS_PARAMETER_CREATE = "mes:process-parameter:create";
    public static final String PROCESS_PARAMETER_UPDATE = "mes:process-parameter:update";
    public static final String PROCESS_PARAMETER_DELETE = "mes:process-parameter:delete";

    // Configuration permissions
    public static final String CONFIG_READ   = "mes:config:read";
    public static final String CONFIG_CREATE = "mes:config:create";
    public static final String CONFIG_UPDATE = "mes:config:update";
    public static final String CONFIG_DELETE = "mes:config:delete";

    // Quality Control permissions
    public static final String QC_INSPECTION_TYPE_READ   = "mes:qc:inspection-type:read";
    public static final String QC_INSPECTION_TYPE_CREATE = "mes:qc:inspection-type:create";
    public static final String QC_INSPECTION_TYPE_UPDATE = "mes:qc:inspection-type:update";
    public static final String QC_INSPECTION_TYPE_DELETE = "mes:qc:inspection-type:delete";
    
    // QC Inspection Plan permissions
    public static final String QC_INSPECTION_PLAN_READ   = "mes:qc:inspection-plan:read";
    public static final String QC_INSPECTION_PLAN_CREATE = "mes:qc:inspection-plan:create";
    public static final String QC_INSPECTION_PLAN_UPDATE = "mes:qc:inspection-plan:update";
    public static final String QC_INSPECTION_PLAN_DELETE = "mes:qc:inspection-plan:delete";
    
    // QC Inspection Item permissions
    public static final String QC_INSPECTION_ITEM_READ   = "mes:qc:inspection-item:read";
    public static final String QC_INSPECTION_ITEM_CREATE = "mes:qc:inspection-item:create";
    public static final String QC_INSPECTION_ITEM_UPDATE = "mes:qc:inspection-item:update";
    public static final String QC_INSPECTION_ITEM_DELETE = "mes:qc:inspection-item:delete";
    
    // QC Defect Code permissions
    public static final String QC_DEFECT_CODE_READ   = "mes:qc:defect-code:read";
    public static final String QC_DEFECT_CODE_CREATE = "mes:qc:defect-code:create";
    public static final String QC_DEFECT_CODE_UPDATE = "mes:qc:defect-code:update";
    public static final String QC_DEFECT_CODE_DELETE = "mes:qc:defect-code:delete";
    
    // QC Trigger Rule permissions
    public static final String QC_TRIGGER_RULE_READ   = "mes:qc:trigger-rule:read";
    public static final String QC_TRIGGER_RULE_CREATE = "mes:qc:trigger-rule:create";
    public static final String QC_TRIGGER_RULE_UPDATE = "mes:qc:trigger-rule:update";
    public static final String QC_TRIGGER_RULE_DELETE = "mes:qc:trigger-rule:delete";
    
    // QC Non-Conformance Disposition permissions
    public static final String QC_NON_CONFORMANCE_READ   = "mes:qc:non-conformance:read";
    public static final String QC_NON_CONFORMANCE_CREATE = "mes:qc:non-conformance:create";
    public static final String QC_NON_CONFORMANCE_UPDATE = "mes:qc:non-conformance:update";
    public static final String QC_NON_CONFORMANCE_DELETE = "mes:qc:non-conformance:delete";
    
    // QC SPC Control Chart permissions
    public static final String QC_SPC_CONTROL_CHART_READ   = "mes:qc:spc-control-chart:read";
    public static final String QC_SPC_CONTROL_CHART_CREATE = "mes:qc:spc-control-chart:create";
    public static final String QC_SPC_CONTROL_CHART_UPDATE = "mes:qc:spc-control-chart:update";
    public static final String QC_SPC_CONTROL_CHART_DELETE = "mes:qc:spc-control-chart:delete";
}
