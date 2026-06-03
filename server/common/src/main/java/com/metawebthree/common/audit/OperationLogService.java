package com.metawebthree.common.audit;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Service for operation log management
 * Uses MyBatis-Plus ServiceImpl for standard CRUD operations
 */
@Service
@Transactional(readOnly = true)
public class OperationLogService extends ServiceImpl<OperationLogRepository, OperationLog> {

    /**
     * Find operation logs by user ID
     */
    public List<OperationLog> findByUserIdOrderByOperationTimeDesc(Long userId) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperationLog::getUserId, userId)
                .orderByDesc(OperationLog::getOperationTime);
        return this.list(wrapper);
    }

    /**
     * Find operation logs by operation type
     */
    public List<OperationLog> findByOperationOrderByOperationTimeDesc(String operation) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperationLog::getOperation, operation)
                .orderByDesc(OperationLog::getOperationTime);
        return this.list(wrapper);
    }

    /**
     * Find operation logs by status
     */
    public List<OperationLog> findByStatusOrderByOperationTimeDesc(String status) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperationLog::getStatus, status)
                .orderByDesc(OperationLog::getOperationTime);
        return this.list(wrapper);
    }

    /**
     * Find operation logs by entity type and entity ID
     */
    public List<OperationLog> findByEntityTypeAndEntityIdOrderByOperationTimeDesc(
            String entityType, Long entityId) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperationLog::getEntityType, entityType)
                .eq(OperationLog::getEntityId, entityId)
                .orderByDesc(OperationLog::getOperationTime);
        return this.list(wrapper);
    }

    /**
     * Find operation logs within time range
     */
    public List<OperationLog> findByOperationTimeBetweenOrderByOperationTimeDesc(
            LocalDateTime startTime, LocalDateTime endTime) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(OperationLog::getOperationTime, startTime, endTime)
                .orderByDesc(OperationLog::getOperationTime);
        return this.list(wrapper);
    }

    /**
     * Find failed operations
     */
    public List<OperationLog> findFailedOperations() {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.eq(OperationLog::getStatus, "FAILURE")
                        .or().eq(OperationLog::getStatus, "ERROR"))
                .orderByDesc(OperationLog::getOperationTime);
        return this.list(wrapper);
    }

    /**
     * Count operations by user ID
     */
    public long countByUserId(Long userId) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperationLog::getUserId, userId);
        return this.count(wrapper);
    }

    /**
     * Count operations by status
     */
    public long countByStatus(String status) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(OperationLog::getStatus, status);
        return this.count(wrapper);
    }

    /**
     * Delete operation logs older than specified date
     */
    @Transactional
    public void deleteOlderThan(LocalDateTime date) {
        LambdaQueryWrapper<OperationLog> wrapper = new LambdaQueryWrapper<>();
        wrapper.lt(OperationLog::getOperationTime, date);
        this.remove(wrapper);
    }
}
