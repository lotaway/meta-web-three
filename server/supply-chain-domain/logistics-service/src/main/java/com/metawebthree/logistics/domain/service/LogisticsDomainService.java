package com.metawebthree.logistics.domain.service;

import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import com.metawebthree.logistics.domain.entity.Carrier;
import java.math.BigDecimal;
import java.util.Optional;

public interface LogisticsDomainService {

    Optional<LogisticsOrder> findByTrackingNo(String trackingNo);

    Optional<Carrier> findCarrierById(Long carrierId);

    String generateTrackingNo(Carrier carrier);

    BigDecimal calculateFreight(Carrier carrier, BigDecimal weight, BigDecimal volume);
}