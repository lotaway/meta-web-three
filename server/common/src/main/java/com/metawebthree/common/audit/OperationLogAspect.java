package com.metawebthree.common.audit;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletRequest;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.util.Arrays;

@Aspect
@Component
public class OperationLogAspect {

    private static final Logger log = LoggerFactory.getLogger(OperationLogAspect.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final int MAX_PARAMS_LENGTH = 2000;
    private static final int MAX_ERROR_MSG_LENGTH = 500;
    private static final String STATUS_SUCCESS = "SUCCESS";
    private static final String STATUS_FAILURE = "FAILURE";
    private static final String UNKNOWN_ERROR = "Unknown error";

    @Autowired
    private OperationLogService operationLogService;

    @Around("@annotation(com.metawebthree.common.annotations.LogMethod)")
    public Object logOperation(ProceedingJoinPoint joinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        com.metawebthree.common.annotations.LogMethod logMethod = method.getAnnotation(
                com.metawebthree.common.annotations.LogMethod.class);

        OperationLog operationLog = initOperationLog(logMethod, signature, method);
        setUserInfoFromAuth(operationLog);
        setRequestInfoFromContext(joinPoint, operationLog);
        extractEntityInfo(joinPoint.getArgs(), operationLog);

        return executeWithLogging(joinPoint, operationLog, startTime);
    }

    private Object executeWithLogging(ProceedingJoinPoint joinPoint, OperationLog operationLog, long startTime) throws Throwable {
        try {
            Object result = joinPoint.proceed();
            operationLog.setStatus(STATUS_SUCCESS);
            return result;
        } catch (Exception e) {
            operationLog.setStatus(STATUS_FAILURE);
            operationLog.setErrorMessage(truncateErrorMessage(e));
            throw e;
        } finally {
            operationLog.setExecutionTime(System.currentTimeMillis() - startTime);
            saveOperationLog(operationLog);
        }
    }

    private OperationLog initOperationLog(com.metawebthree.common.annotations.LogMethod logMethod,
                                           MethodSignature signature, Method method) {
        OperationLog operationLog = new OperationLog();
        operationLog.setOperationTime(LocalDateTime.now());
        operationLog.setOperation(logMethod.value());
        operationLog.setMethod(signature.getDeclaringTypeName() + "." + method.getName());
        return operationLog;
    }

    private void setUserInfoFromAuth(OperationLog operationLog) {
        try {
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            if (authentication != null && authentication.isAuthenticated()) {
                String username = authentication.getName();
                operationLog.setUsername(username);
                if (authentication.getDetails() != null) {
                    operationLog.setUserId(tryExtractUserId(authentication));
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract user info", e);
        }
    }

    private void setRequestInfoFromContext(ProceedingJoinPoint joinPoint, OperationLog operationLog) {
        try {
            ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            if (attributes != null) {
                HttpServletRequest request = attributes.getRequest();
                operationLog.setIp(getClientIp(request));
                try {
                    String params = objectMapper.writeValueAsString(joinPoint.getArgs());
                    operationLog.setParams(params.length() > MAX_PARAMS_LENGTH ? params.substring(0, MAX_PARAMS_LENGTH) : params);
                } catch (JsonProcessingException e) {
                    operationLog.setParams(Arrays.toString(joinPoint.getArgs()));
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract request info", e);
        }
    }

    private String truncateErrorMessage(Exception e) {
        String msg = e.getMessage();
        if (msg == null) return UNKNOWN_ERROR;
        return msg.length() > MAX_ERROR_MSG_LENGTH ? msg.substring(0, MAX_ERROR_MSG_LENGTH) : msg;
    }

    private void saveOperationLog(OperationLog operationLog) {
        try {
            operationLogService.save(operationLog);
        } catch (Exception e) {
            log.error("Failed to save operation log", e);
        }
    }

    private Long tryExtractUserId(Authentication authentication) {
        return null;
    }

    private void extractEntityInfo(Object[] args, OperationLog operationLog) {
        if (args == null || args.length == 0) {
            return;
        }
        for (Object arg : args) {
            if (trySetEntityInfo(arg, operationLog)) {
                break;
            }
        }
    }

    private boolean trySetEntityInfo(Object arg, OperationLog operationLog) {
        if (arg == null) {
            return false;
        }
        try {
            Method getIdMethod = arg.getClass().getMethod("getId");
            Object id = getIdMethod.invoke(arg);
            if (id != null) {
                operationLog.setEntityType(arg.getClass().getSimpleName());
                operationLog.setEntityId(Long.valueOf(id.toString()));
                return true;
            }
        } catch (NoSuchMethodException e) {
            log.trace("Argument {} has no getId() method", arg.getClass().getName());
        } catch (Exception e) {
            log.warn("Error extracting entity info from {}", arg.getClass().getName(), e);
        }
        return false;
    }

    private String getClientIp(HttpServletRequest request) {
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
