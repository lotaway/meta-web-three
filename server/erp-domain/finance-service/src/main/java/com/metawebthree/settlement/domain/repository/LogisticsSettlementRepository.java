package com.metawebthree.settlement.domain.repository;

import com.metawebthree.settlement.domain.entity.LogisticsSettlement;
import java.util.List;
import java.util.Optional;

public interface LogisticsSettlementRepository {
    LogisticsSettlement save(LogisticsSettlement settlement);
    Optional<LogisticsSettlement> findById(Long id);
    Optional<LogisticsSettlement> findByTrackingNo(String trackingNo);
    List<LogisticsSettlement> findByCarrierId(Long carrierId);
    List<LogisticsSettlement> findByStatus(LogisticsSettlement.LogisticsSettlementStatus status);
    List<LogisticsSettlement> findAll();
}