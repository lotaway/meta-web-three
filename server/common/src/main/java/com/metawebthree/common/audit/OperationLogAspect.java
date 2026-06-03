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

/**
 * AOP aspect for automatic operation logging
 * Records who operated on what data at what time
 */
@Aspect
@Component
public class OperationLogAspect {

    private static final Logger log = LoggerFactory.getLogger(OperationLogAspect.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    @Autowired
    private OperationLogService operationLogService;

    @Around("@annotation(com.metawebthree.common.annotations.LogMethod)")
    public Object logOperation(ProceedingJoinPoint joinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        com.metawebthree.common.annotations.LogMethod logMethod = method.getAnnotation(
                com.metawebthree.common.annotations.LogMethod.class);

        OperationLog operationLog = new OperationLog();
        operationLog.setOperationTime(LocalDateTime.now());
        operationLog.setOperation(logMethod.value());
        operationLog.setMethod(signature.getDeclaringTypeName() + "." + method.getName());

        // Get current user info
        try {
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            if (authentication != null && authentication.isAuthenticated()) {
                String username = authentication.getName();
                operationLog.setUsername(username);
                // Try to extract userId from authentication details if available
                if (authentication.getDetails() != null) {
                    operationLog.setUserId(tryExtractUserId(authentication));
                }
            }
        } catch (Exception e) {
            log.warn("Failed to extract user info: {}", e.getMessage());
        }

        // Get request info
        try {
            ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            if (attributes != null) {
                HttpServletRequest request = attributes.getRequest();
                operationLog.setIp(getClientIp(request));
                // Log request parameters
                try {
                    String params = objectMapper.writeValueAsString(joinPoint.getArgs());
                    operationLog.setParams(params.length() > 2000 ? params.substring(0, 2000) : params);
                } catch (JsonProcessingException e) {
                    operationLog.setParams(Arrays.toString(joinPoint.getArgs()));
                }
            }
        } catch (Exception e) {
            log.warn("Failed to extract request info: {}", e.getMessage());
        }

        // Extract entity info from method parameters if available
        extractEntityInfo(joinPoint.getArgs(), operationLog);

        Object result = null;
        try {
            result = joinPoint.proceed();
            operationLog.setStatus("SUCCESS");
            return result;
        } catch (Exception e) {
            operationLog.setStatus("FAILURE");
            operationLog.setErrorMessage(e.getMessage() != null ?
                    e.getMessage().length() > 500 ? e.getMessage().substring(0, 500) : e.getMessage()
                    : "Unknown error");
            throw e;
        } finally {
            long endTime = System.currentTimeMillis();
            operationLog.setExecutionTime(endTime - startTime);

            // Save operation log asynchronously to avoid affecting main business logic
            try {
                operationLogService.save(operationLog);
            } catch (Exception e) {
                log.error("Failed to save operation log: {}", e.getMessage());
            }
        }
    }

    /**
     * Extract user ID from authentication
     */
    private Long tryExtractUserId(Authentication authentication) {
        // This can be customized based on actual authentication structure
        // For example, if Principal is a custom UserDetails implementation
        Object principal = authentication.getPrincipal();
        if (principal instanceof org.springframework.security.core.userdetails.UserDetails) {
            // Custom logic to extract userId from UserDetails
            // This depends on actual implementation
        }
        return null; // Return null if cannot extract
    }

    /**
     * Extract entity type and ID from method parameters
     */
    private void extractEntityInfo(Object[] args, OperationLog operationLog) {
        if (args == null || args.length == 0) {
            return;
        }

        for (Object arg : args) {
            if (arg == null) {
                continue;
            }

            // Check if argument has getId() method (common for entities)
            try {
                Method getIdMethod = arg.getClass().getMethod("getId");
                if (getIdMethod != null) {
                    Object id = getIdMethod.invoke(arg);
                    if (id != null) {
                        operationLog.setEntityType(arg.getClass().getSimpleName());
                        operationLog.setEntityId(Long.valueOf(id.toString()));
                        break;
                    }
                }
            } catch (Exception e) {
                // Ignore, argument doesn't have getId() method
            }
        }
    }

    /**
     * Get client IP address
     */
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
