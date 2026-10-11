package com.metawebthree.logistics.domain.service;

import com.metawebthree.logistics.domain.entity.LogisticsOrder;
import com.metawebthree.logistics.domain.entity.Carrier;
import com.metawebthree.logistics.infrastructure.persistence.repository.CarrierRepository;
import com.metawebthree.logistics.infrastructure.persistence.repository.LogisticsOrderRepository;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class LogisticsDomainServiceImpl implements LogisticsDomainService {

    private final LogisticsOrderRepository logisticsOrderRepository;
    private final CarrierRepository carrierRepository;
    private final AtomicLong trackingNoSeq = new AtomicLong(1000);

    public LogisticsDomainServiceImpl(LogisticsOrderRepository logisticsOrderRepository,
                                        CarrierRepository carrierRepository) {
        this.logisticsOrderRepository = logisticsOrderRepository;
        this.carrierRepository = carrierRepository;
    }

    @Override
    public Optional<LogisticsOrder> findByTrackingNo(String trackingNo) {
        return logisticsOrderRepository.findByTrackingNo(trackingNo);
    }

    @Override
    public Optional<Carrier> findCarrierById(Long carrierId) {
        return carrierRepository.findById(carrierId);
    }

    @Override
    public String generateTrackingNo(Carrier carrier) {
        return carrier.getCarrierCode() + System.currentTimeMillis() + 
            trackingNoSeq.getAndIncrement();
    }

    @Override
    public BigDecimal calculateFreight(Carrier carrier, BigDecimal weight, BigDecimal volume) {
        BigDecimal weightCost = weight.multiply(carrier.getWeightUnitPrice());
        BigDecimal volumeCost = volume.multiply(carrier.getVolumeUnitPrice());
        return carrier.getBaseFreight().add(weightCost).add(volumeCost);
    }
}