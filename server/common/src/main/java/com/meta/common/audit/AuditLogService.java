package com.meta.common.audit;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import jakarta.servlet.http.HttpServletRequest;
import java.time.LocalDateTime;
import java.util.List;

@Service
public class AuditLogService {

    private static final Logger logger = LoggerFactory.getLogger(AuditLogService.class);

    @Autowired
    private AuditLogRepository auditLogRepository;

    @Autowired(required = false)
    private HttpServletRequest httpServletRequest;

    public AuditLog log(AuditLog auditLog) {
        try {
            if (httpServletRequest != null) {
                if (auditLog.getIpAddress() == null) {
                    auditLog.setIpAddress(getClientIpAddress(httpServletRequest));
                }
                if (auditLog.getRequestUrl() == null) {
                    auditLog.setRequestUrl(httpServletRequest.getRequestURI());
                }
                if (auditLog.getRequestMethod() == null) {
                    auditLog.setRequestMethod(httpServletRequest.getMethod());
                }
            }

            if (auditLog.getOperationTime() == null) {
                auditLog.setOperationTime(LocalDateTime.now());
            }

            auditLogRepository.insert(auditLog);
            logger.debug("Audit log saved: {}", auditLog);
            return auditLog;
        } catch (Exception e) {
            logger.error("Failed to save audit log", e);
            throw new RuntimeException("Failed to save audit log", e);
        }
    }

    public AuditLog log(String operationType, String resourceType, String resourceId,
                       String description, String username, Long userId) {
        AuditLog auditLog = new AuditLog();
        auditLog.setOperationType(operationType);
        auditLog.setResourceType(resourceType);
        auditLog.setResourceId(resourceId);
        auditLog.setDescription(description);
        auditLog.setUsername(username);
        auditLog.setUserId(userId);
        auditLog.setResult("SUCCESS");

        return log(auditLog);
    }

    public AuditLog logFailure(String operationType, String resourceType, String resourceId,
                              String description, String errorMessage, String username, Long userId) {
        AuditLog auditLog = new AuditLog();
        auditLog.setOperationType(operationType);
        auditLog.setResourceType(resourceType);
        auditLog.setResourceId(resourceId);
        auditLog.setDescription(description);
        auditLog.setResult("FAILURE");
        auditLog.setErrorMessage(errorMessage);
        auditLog.setUsername(username);
        auditLog.setUserId(userId);

        return log(auditLog);
    }

    public AuditLog findById(Long id) {
        return auditLogRepository.selectById(id);
    }

    public List<AuditLog> findAll() {
        return auditLogRepository.selectList(null);
    }

    public List<AuditLog> findByCondition(AuditLogQueryCondition condition) {
        QueryWrapper<AuditLog> wrapper = new QueryWrapper<>();
        if (condition.getUserId() != null) {
            wrapper.eq("user_id", condition.getUserId());
        }
        if (condition.getUsername() != null) {
            wrapper.eq("username", condition.getUsername());
        }
        if (condition.getOperationType() != null) {
            wrapper.eq("operation_type", condition.getOperationType());
        }
        if (condition.getResourceType() != null) {
            wrapper.eq("resource_type", condition.getResourceType());
        }
        if (condition.getResourceId() != null) {
            wrapper.eq("resource_id", condition.getResourceId());
        }
        if (condition.getResult() != null) {
            wrapper.eq("result", condition.getResult());
        }
        if (condition.getStartTime() != null) {
            wrapper.ge("operation_time", condition.getStartTime());
        }
        if (condition.getEndTime() != null) {
            wrapper.le("operation_time", condition.getEndTime());
        }
        if (condition.getIpAddress() != null) {
            wrapper.eq("ip_address", condition.getIpAddress());
        }
        if (condition.getRequestUrl() != null) {
            wrapper.eq("request_url", condition.getRequestUrl());
        }
        wrapper.orderByDesc("operation_time");
        return auditLogRepository.selectList(wrapper);
    }

    public List<AuditLog> findByTimeRange(LocalDateTime startTime, LocalDateTime endTime) {
        return auditLogRepository.findByOperationTimeBetweenOrderByOperationTimeDesc(startTime, endTime);
    }

    public List<AuditLog> findByUsername(String username) {
        return auditLogRepository.findByUsernameOrderByOperationTimeDesc(username);
    }

    public List<AuditLog> findByOperationType(String operationType) {
        return auditLogRepository.findByOperationTypeOrderByOperationTimeDesc(operationType);
    }

    public List<AuditLog> findByResourceType(String resourceType) {
        return auditLogRepository.findByResourceTypeOrderByOperationTimeDesc(resourceType);
    }

    public int cleanupBefore(LocalDateTime time) {
        return auditLogRepository.deleteBefore(time);
    }

    public long count() {
        return auditLogRepository.selectCount(null);
    }

    public void clearAll() {
        auditLogRepository.delete(null);
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
