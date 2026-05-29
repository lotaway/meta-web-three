package com.metawebthree.logistics.infrastructure.persistence.repository;

import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import java.util.List;
import java.util.Optional;

public interface LogisticsOrderRepository {

    Optional<LogisticsOrder> findById(Long id);

    Optional<LogisticsOrder> findByTrackingNo(String trackingNo);

    Optional<LogisticsOrder> findByOrderNo(String orderNo);

    List<LogisticsOrder> findAll();

    List<LogisticsOrder> findByCarrierId(Long carrierId);

    List<LogisticsOrder> findByStatus(String status);

    List<LogisticsOrder> findByCarrierIdAndStatus(Long carrierId, String status);

    LogisticsOrder save(LogisticsOrder order);

    void delete(LogisticsOrder order);
}