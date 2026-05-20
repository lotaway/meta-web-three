package com.metawebthree.settlement.domain.repository;

import com.metawebthree.settlement.domain.entity.SettlementOrder;
import java.util.List;
import java.util.Optional;

public interface SettlementOrderRepository {
    Optional<SettlementOrder> findById(Long id);
    Optional<SettlementOrder> findBySettlementNo(String settlementNo);
    List<SettlementOrder> findByStatus(SettlementOrder.SettlementStatus status);
    List<SettlementOrder> findByMerchantId(Long merchantId);
    List<SettlementOrder> findByDateRange(java.time.LocalDateTime start, java.time.LocalDateTime end);
    List<SettlementOrder> findAll();
    void save(SettlementOrder order);
    void update(SettlementOrder order);
    void delete(Long id);
}