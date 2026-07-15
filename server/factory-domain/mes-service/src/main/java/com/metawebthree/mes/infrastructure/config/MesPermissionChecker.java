package com.metawebthree.mes.infrastructure.config;

import com.metawebthree.common.annotations.PermissionChecker;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.mes.common.MesPermissions;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.util.Map;
import java.util.Set;

@Component
public class MesPermissionChecker implements PermissionChecker {

    private static final Map<String, Set<String>> ROLE_PERMISSIONS = Map.of(
        "ADMIN", Set.of(
            MesPermissions.WORK_ORDER_READ, MesPermissions.WORK_ORDER_CREATE, MesPermissions.WORK_ORDER_UPDATE, MesPermissions.WORK_ORDER_DELETE, MesPermissions.WORK_ORDER_RELEASE,
            MesPermissions.TASK_READ, MesPermissions.TASK_CREATE, MesPermissions.TASK_UPDATE, MesPermissions.TASK_START, MesPermissions.TASK_COMPLETE,
            MesPermissions.EQUIPMENT_READ, MesPermissions.EQUIPMENT_CREATE, MesPermissions.EQUIPMENT_UPDATE, MesPermissions.EQUIPMENT_DELETE, MesPermissions.EQUIPMENT_BREAKDOWN, MesPermissions.EQUIPMENT_REPAIR,
            MesPermissions.PROCESS_ROUTE_READ, MesPermissions.PROCESS_ROUTE_CREATE, MesPermissions.PROCESS_ROUTE_UPDATE, MesPermissions.PROCESS_ROUTE_DELETE,
            MesPermissions.PROCESS_PARAMETER_READ, MesPermissions.PROCESS_PARAMETER_CREATE, MesPermissions.PROCESS_PARAMETER_UPDATE, MesPermissions.PROCESS_PARAMETER_DELETE,
            MesPermissions.CONFIG_READ, MesPermissions.CONFIG_CREATE, MesPermissions.CONFIG_UPDATE, MesPermissions.CONFIG_DELETE,
            MesPermissions.SCADA_READ, MesPermissions.SCADA_TELEMETRY_READ, MesPermissions.SCADA_COMMAND_DISPATCH, MesPermissions.SCADA_COMMAND_READ,
            MesPermissions.TRACE_READ, MesPermissions.TRACE_FORWARD, MesPermissions.TRACE_BACKWARD, MesPermissions.TRACE_CHAIN, MesPermissions.TRACE_CREATE, MesPermissions.TRACE_UPDATE, MesPermissions.TRACE_DELETE,
            MesPermissions.SCHEDULING_READ, MesPermissions.SCHEDULING_CREATE, MesPermissions.SCHEDULING_UPDATE, MesPermissions.SCHEDULING_DELETE, MesPermissions.SCHEDULING_FORWARD, MesPermissions.SCHEDULING_BACKWARD,
            MesPermissions.LABOR_OPERATOR_READ, MesPermissions.LABOR_OPERATOR_CREATE, MesPermissions.LABOR_OPERATOR_UPDATE, MesPermissions.LABOR_OPERATOR_DELETE,
            MesPermissions.LABOR_ATTENDANCE_READ, MesPermissions.LABOR_ATTENDANCE_CREATE,
            MesPermissions.LABOR_TIME_RECORD_READ, MesPermissions.LABOR_TIME_RECORD_CREATE,
            MesPermissions.LABOR_ASSIGNMENT_READ, MesPermissions.LABOR_ASSIGNMENT_CREATE
        ),
        "OPERATOR", Set.of(
            MesPermissions.WORK_ORDER_READ, MesPermissions.WORK_ORDER_CREATE, MesPermissions.WORK_ORDER_UPDATE, MesPermissions.WORK_ORDER_RELEASE,
            MesPermissions.TASK_READ, MesPermissions.TASK_CREATE, MesPermissions.TASK_UPDATE, MesPermissions.TASK_START, MesPermissions.TASK_COMPLETE,
            MesPermissions.EQUIPMENT_READ, MesPermissions.EQUIPMENT_UPDATE, MesPermissions.EQUIPMENT_BREAKDOWN, MesPermissions.EQUIPMENT_REPAIR,
            MesPermissions.PROCESS_ROUTE_READ,
            MesPermissions.PROCESS_PARAMETER_READ, MesPermissions.PROCESS_PARAMETER_CREATE, MesPermissions.PROCESS_PARAMETER_UPDATE,
            MesPermissions.CONFIG_READ,
            MesPermissions.SCADA_READ, MesPermissions.SCADA_TELEMETRY_READ, MesPermissions.SCADA_COMMAND_READ,
            MesPermissions.TRACE_READ, MesPermissions.TRACE_FORWARD, MesPermissions.TRACE_BACKWARD, MesPermissions.TRACE_CHAIN,
            MesPermissions.SCHEDULING_READ,
            MesPermissions.LABOR_OPERATOR_READ, MesPermissions.LABOR_ATTENDANCE_READ, MesPermissions.LABOR_TIME_RECORD_READ
        ),
        "VIEWER", Set.of(
            MesPermissions.WORK_ORDER_READ,
            MesPermissions.TASK_READ,
            MesPermissions.EQUIPMENT_READ,
            MesPermissions.PROCESS_ROUTE_READ,
            MesPermissions.PROCESS_PARAMETER_READ,
            MesPermissions.CONFIG_READ,
            MesPermissions.SCADA_READ, MesPermissions.SCADA_TELEMETRY_READ, MesPermissions.SCADA_COMMAND_READ,
            MesPermissions.TRACE_READ, MesPermissions.TRACE_FORWARD, MesPermissions.TRACE_BACKWARD, MesPermissions.TRACE_CHAIN,
            MesPermissions.SCHEDULING_READ,
            MesPermissions.LABOR_OPERATOR_READ, MesPermissions.LABOR_ATTENDANCE_READ, MesPermissions.LABOR_TIME_RECORD_READ
        )
    );

    @Override
    public boolean hasPermission(Long userId, String permission) {
        String role = getRoleFromRequest();
        if (role == null) {
            return false;
        }
        Set<String> permissions = ROLE_PERMISSIONS.get(role.toUpperCase());
        return permissions != null && permissions.contains(permission);
    }

    private String getRoleFromRequest() {
        ServletRequestAttributes attrs = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attrs == null) {
            return null;
        }
        HttpServletRequest request = attrs.getRequest();
        String role = request.getHeader(HeaderConstants.USER_ROLE);
        return role;
    }
}
