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
}
