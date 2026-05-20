package com.metawebthree.aiwarehouse.domain.repository;

import com.metawebthree.aiwarehouse.domain.entity.AIRequestRecord;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface AIRequestRecordRepository {
    AIRequestRecord save(AIRequestRecord record);
    Optional<AIRequestRecord> findById(Long id);
    Optional<AIRequestRecord> findByRequestId(String requestId);
    List<AIRequestRecord> findByCapabilityId(String capabilityId);
    List<AIRequestRecord> findByCallerServiceName(String serviceName);
    List<AIRequestRecord> findByStatus(AIRequestRecord.AIRequestStatus status);
    List<AIRequestRecord> findByCreatedAtBetween(LocalDateTime start, LocalDateTime end);
    List<AIRequestRecord> findTop100ByOrderByCreatedAtDesc();
    void deleteById(Long id);
    long count();
}