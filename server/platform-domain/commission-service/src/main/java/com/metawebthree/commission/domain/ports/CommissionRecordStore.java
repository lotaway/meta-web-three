package com.metawebthree.commission.domain.ports;

import java.time.LocalDateTime;
import java.util.List;

import com.metawebthree.commission.domain.CommissionRecord;

public interface CommissionRecordStore {
    long countActiveByOrderId(Long orderId);
    void save(CommissionRecord record);
    List<CommissionRecord> findPendingBefore(LocalDateTime executeBefore);
    List<CommissionRecord> findActiveByOrderId(Long orderId);
    boolean updateStatus(Long id, String fromStatus, String toStatus, LocalDateTime updatedAt);
    boolean updateStatus(Long id, String toStatus, LocalDateTime updatedAt);
    List<CommissionRecord> findByUserId(Long userId, String status, int page, int size);
}
