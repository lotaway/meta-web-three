package com.metawebthree.common;

public final class MesPermissions {

    private MesPermissions() {}

    // SCADA 权限
    public static final String SCADA_TELEMETRY_READ = "mes:scada:telemetry:read";
    public static final String SCADA_COMMAND_DISPATCH = "mes:scada:command:dispatch";
    public static final String SCADA_COMMAND_READ = "mes:scada:command:read";

    // 追溯权限
    public static final String TRACE_READ = "mes:trace:read";
    public static final String TRACE_FORWARD = "mes:trace:forward";
    public static final String TRACE_BACKWARD = "mes:trace:backward";
    public static final String TRACE_CHAIN = "mes:trace:chain";
    public static final String TRACE_CREATE = "mes:trace:create";
    public static final String TRACE_UPDATE = "mes:trace:update";
    public static final String TRACE_DELETE = "mes:trace:delete";

    // 排程权限
    public static final String SCHEDULING_READ = "mes:scheduling:read";
    public static final String SCHEDULING_CREATE = "mes:scheduling:create";
    public static final String SCHEDULING_UPDATE = "mes:scheduling:update";
    public static final String SCHEDULING_DELETE = "mes:scheduling:delete";
    public static final String SCHEDULING_FORWARD = "mes:scheduling:forward";
    public static final String SCHEDULING_BACKWARD = "mes:scheduling:backward";

    // 人员工时权限
    public static final String LABOR_OPERATOR_READ = "mes:labor:operator:read";
    public static final String LABOR_OPERATOR_CREATE = "mes:labor:operator:create";
    public static final String LABOR_OPERATOR_UPDATE = "mes:labor:operator:update";
    public static final String LABOR_OPERATOR_DELETE = "mes:labor:operator:delete";
    public static final String LABOR_ATTENDANCE_READ = "mes:labor:attendance:read";
    public static final String LABOR_ATTENDANCE_CREATE = "mes:labor:attendance:create";
    public static final String LABOR_TIME_RECORD_READ = "mes:labor:time-record:read";
    public static final String LABOR_TIME_RECORD_CREATE = "mes:labor:time-record:create";
    public static final String LABOR_ASSIGNMENT_READ = "mes:labor:assignment:read";
    public static final String LABOR_ASSIGNMENT_CREATE = "mes:labor:assignment:create";
}