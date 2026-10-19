package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.ReservationRecord;
import org.springframework.stereotype.Repository;
import java.util.Optional;

@Repository
public interface ReservationRepository {
    Optional<ReservationRecord> findByBizId(String bizId);
    ReservationRecord save(ReservationRecord record);
    void delete(ReservationRecord record);
    boolean compareAndSetStatus(Long id, String expectedStatus, String nextStatus);
}