package com.meta.common.audit;

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

            AuditLog saved = auditLogRepository.save(auditLog);
            logger.debug("Audit log saved: {}", saved);
            return saved;
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
        return auditLogRepository.findById(id);
    }

    public List<AuditLog> findAll() {
        return auditLogRepository.findAll();
    }

    public List<AuditLog> findByCondition(AuditLogQueryCondition condition) {
        return auditLogRepository.findByCondition(condition);
    }

    public List<AuditLog> findByTimeRange(LocalDateTime startTime, LocalDateTime endTime) {
        return auditLogRepository.findByTimeRange(startTime, endTime);
    }

    public List<AuditLog> findByUsername(String username) {
        return auditLogRepository.findByUsername(username);
    }

    public List<AuditLog> findByOperationType(String operationType) {
        return auditLogRepository.findByOperationType(operationType);
    }

    public List<AuditLog> findByResourceType(String resourceType) {
        return auditLogRepository.findByResourceType(resourceType);
    }

    public int cleanupBefore(LocalDateTime time) {
        return auditLogRepository.deleteBefore(time);
    }

    public long count() {
        return auditLogRepository.count();
    }

    public void clearAll() {
        auditLogRepository.deleteAll();
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
