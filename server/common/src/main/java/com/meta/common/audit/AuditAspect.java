package com.meta.common.audit;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.DefaultParameterNameDiscoverer;
import org.springframework.core.ParameterNameDiscoverer;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import java.util.HashMap;
import java.util.Map;

@Aspect
@Component
public class AuditAspect {

    private static final Logger logger = LoggerFactory.getLogger(AuditAspect.class);
    private static final String STATUS_SUCCESS = "SUCCESS";
    private static final String STATUS_FAILURE = "FAILURE";
    private static final String ARG_PREFIX = "arg";
    private static final String EMPTY_JSON = "{}";

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ParameterNameDiscoverer parameterNameDiscoverer = new DefaultParameterNameDiscoverer();

    @Autowired
    private AuditLogService auditLogService;

    @Around("@annotation(audit)")
    public Object audit(ProceedingJoinPoint joinPoint, Audit audit) throws Throwable {
        long startTime = System.currentTimeMillis();
        AuditLog auditLog = initAuditLog(audit);

        try {
            setUserInfo(auditLog);
            setRequestInfo(auditLog);
            recordRequestParams(joinPoint, audit, auditLog);
            Object result = joinPoint.proceed();
            recordResponseData(result, audit, auditLog);
            completeAuditSuccess(auditLog, startTime);
            return result;
        } catch (Exception e) {
            handleAuditFailure(auditLog, e, startTime);
            throw e;
        }
    }

    private AuditLog initAuditLog(Audit audit) {
        AuditLog auditLog = new AuditLog();
        auditLog.setOperationType(audit.operationType());
        auditLog.setResourceType(audit.resourceType());
        auditLog.setDescription(audit.description());
        return auditLog;
    }

    private void recordRequestParams(ProceedingJoinPoint joinPoint, Audit audit, AuditLog auditLog) {
        if (!audit.recordParams()) return;
        String params = getMethodParams(joinPoint, audit.resourceIdParam());
        auditLog.setRequestParams(params);
        if (!audit.resourceIdParam().isEmpty()) {
            String resourceId = extractResourceId(joinPoint, audit.resourceIdParam());
            auditLog.setResourceId(resourceId);
        }
    }

    private void recordResponseData(Object result, Audit audit, AuditLog auditLog) {
        if (!audit.recordResponse()) return;
        try {
            String responseData = objectMapper.writeValueAsString(result);
            if (audit.ignoreSensitive()) {
                responseData = maskSensitiveData(responseData);
            }
            auditLog.setResponseData(responseData);
        } catch (JsonProcessingException e) {
            logger.error("Failed to serialize response data", e);
        }
    }

    private void completeAuditSuccess(AuditLog auditLog, long startTime) {
        auditLog.setResult(STATUS_SUCCESS);
        auditLog.setDuration(System.currentTimeMillis() - startTime);
        auditLogService.log(auditLog);
    }

    private void handleAuditFailure(AuditLog auditLog, Exception e, long startTime) {
        auditLog.setResult(STATUS_FAILURE);
        auditLog.setErrorMessage(e.getMessage());
        auditLog.setDuration(System.currentTimeMillis() - startTime);
        try {
            auditLogService.log(auditLog);
        } catch (Exception logError) {
            logger.error("Failed to save audit log", logError);
        }
    }

    private void setUserInfo(AuditLog auditLog) {
        try {
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            if (authentication != null && authentication.isAuthenticated()) {
                String username = authentication.getName();
                auditLog.setUsername(username);
            }
        } catch (Exception e) {
            logger.error("Failed to get user info", e);
        }
    }

    private void setRequestInfo(AuditLog auditLog) {
        try {
            ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            if (attributes != null) {
                HttpServletRequest request = attributes.getRequest();
                auditLog.setIpAddress(getClientIpAddress(request));
                auditLog.setRequestUrl(request.getRequestURI());
                auditLog.setRequestMethod(request.getMethod());
            }
        } catch (Exception e) {
            logger.error("Failed to get request info", e);
        }
    }

    private String getMethodParams(ProceedingJoinPoint joinPoint, String resourceIdParam) {
        try {
            Object[] args = joinPoint.getArgs();
            if (args == null || args.length == 0) {
                return EMPTY_JSON;
            }
            Map<String, Object> paramsMap = buildParamsMap(joinPoint, args, resourceIdParam);
            return objectMapper.writeValueAsString(paramsMap);
        } catch (Exception e) {
            logger.error("Failed to get method params", e);
            return EMPTY_JSON;
        }
    }

    private Map<String, Object> buildParamsMap(ProceedingJoinPoint joinPoint, Object[] args, String resourceIdParam) {
        Map<String, Object> paramsMap = new HashMap<>();
        String[] paramNames = parameterNameDiscoverer.getParameterNames(((MethodSignature) joinPoint.getSignature()).getMethod());
        for (int i = 0; i < args.length; i++) {
            if (args[i] != null) {
                if (!resourceIdParam.isEmpty() && paramNames != null && paramNames[i].equals(resourceIdParam)) {
                    continue;
                }
                String paramName = paramNames != null && paramNames[i] != null ? paramNames[i] : ARG_PREFIX + i;
                paramsMap.put(paramName, args[i]);
            }
        }
        return paramsMap;
    }

    private String extractResourceId(ProceedingJoinPoint joinPoint, String resourceIdParam) {
        try {
            Object[] args = joinPoint.getArgs();
            String[] paramNames = parameterNameDiscoverer.getParameterNames(((MethodSignature) joinPoint.getSignature()).getMethod());

            if (paramNames != null) {
                for (int i = 0; i < paramNames.length; i++) {
                    if (paramNames[i].equals(resourceIdParam) && i < args.length) {
                        return args[i] != null ? args[i].toString() : null;
                    }
                }
            }
        } catch (Exception e) {
            logger.error("Failed to extract resource ID", e);
        }
        return null;
    }

    private String maskSensitiveData(String data) {
        return data;
    }

    private String getClientIpAddress(HttpServletRequest request) {
        String[] headerNames = {
            "X-Forwarded-For",
            "Proxy-Client-IP",
            "WL-Proxy-Client-IP",
            "HTTP_CLIENT_IP",
            "HTTP_X_FORWARDED_FOR"
        };

        for (String header : headerNames) {
            String ip = request.getHeader(header);
            if (ip != null && ip.length() > 0 && !"unknown".equalsIgnoreCase(ip)) {
                return ip.split(",")[0].trim();
            }
        }

        return request.getRemoteAddr();
    }
}
