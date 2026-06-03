package com.meta.common.audit;

import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Repository
public class AuditLogRepository {

    private final ConcurrentHashMap<Long, AuditLog> auditLogMap = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);

    public AuditLog save(AuditLog auditLog) {
        if (auditLog.getId() == null) {
            auditLog.setId(idGenerator.getAndIncrement());
        }
        auditLogMap.put(auditLog.getId(), auditLog);
        return auditLog;
    }

    public AuditLog findById(Long id) {
        return auditLogMap.get(id);
    }

    public List<AuditLog> findAll() {
        return new ArrayList<>(auditLogMap.values());
    }

    public List<AuditLog> findByCondition(AuditLogQueryCondition condition) {
        return auditLogMap.values().stream()
                .filter(log -> condition.matches(log))
                .sorted((a, b) -> b.getOperationTime().compareTo(a.getOperationTime()))
                .collect(java.util.stream.Collectors.toList());
    }

    public List<AuditLog> findByTimeRange(LocalDateTime startTime, LocalDateTime endTime) {
        return auditLogMap.values().stream()
                .filter(log -> !log.getOperationTime().isBefore(startTime)
                        && !log.getOperationTime().isAfter(endTime))
                .sorted((a, b) -> b.getOperationTime().compareTo(a.getOperationTime()))
                .collect(java.util.stream.Collectors.toList());
    }

    public List<AuditLog> findByUsername(String username) {
        return auditLogMap.values().stream()
                .filter(log -> username.equals(log.getUsername()))
                .sorted((a, b) -> b.getOperationTime().compareTo(a.getOperationTime()))
                .collect(java.util.stream.Collectors.toList());
    }

    public List<AuditLog> findByOperationType(String operationType) {
        return auditLogMap.values().stream()
                .filter(log -> operationType.equals(log.getOperationType()))
                .sorted((a, b) -> b.getOperationTime().compareTo(a.getOperationTime()))
                .collect(java.util.stream.Collectors.toList());
    }

    public List<AuditLog> findByResourceType(String resourceType) {
        return auditLogMap.values().stream()
                .filter(log -> resourceType.equals(log.getResourceType()))
                .sorted((a, b) -> b.getOperationTime().compareTo(a.getOperationTime()))
                .collect(java.util.stream.Collectors.toList());
    }

    public int deleteBefore(LocalDateTime time) {
        List<Long> toRemove = auditLogMap.values().stream()
                .filter(log -> log.getOperationTime().isBefore(time))
                .map(AuditLog::getId)
                .collect(java.util.stream.Collectors.toList());

        toRemove.forEach(auditLogMap::remove);
        return toRemove.size();
    }

    public long count() {
        return auditLogMap.size();
    }

    public void deleteAll() {
        auditLogMap.clear();
        idGenerator.set(1);
    }
}
