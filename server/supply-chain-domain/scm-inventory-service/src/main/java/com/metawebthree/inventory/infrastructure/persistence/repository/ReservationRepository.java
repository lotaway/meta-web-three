package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.ReservationRecord;
import java.util.Optional;

public interface ReservationRepository {
    Optional<ReservationRecord> findByBizId(String bizId);
    ReservationRecord save(ReservationRecord record);
    void delete(ReservationRecord record);
}