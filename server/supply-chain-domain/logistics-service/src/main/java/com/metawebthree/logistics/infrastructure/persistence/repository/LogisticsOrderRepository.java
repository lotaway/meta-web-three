package com.metawebthree.logistics.infrastructure.persistence.repository;

import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import java.util.Optional;

public interface LogisticsOrderRepository {

    Optional<LogisticsOrder> findByTrackingNo(String trackingNo);

    Optional<LogisticsOrder> findByOrderNo(String orderNo);

    LogisticsOrder save(LogisticsOrder order);
}