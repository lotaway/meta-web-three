package com.metawebthree.settlement.application.command;

import com.metawebthree.settlement.domain.entity.SettlementOrder;
import com.metawebthree.settlement.domain.repository.SettlementOrderRepository;
import com.metawebthree.settlement.infrastructure.event.SettlementEventPublisher;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;

@Service
public class SettlementCommandService {
    private final SettlementOrderRepository repository;
    private final SettlementEventPublisher eventPublisher;

    public SettlementCommandService(SettlementOrderRepository repository, SettlementEventPublisher eventPublisher) {
        this.repository = repository;
        this.eventPublisher = eventPublisher;
    }

    public Long createSettlement(String settlementNo, String orderNo, Long merchantId, 
                                  String merchantName, BigDecimal orderAmount, BigDecimal commissionRate) {
        SettlementOrder order = new SettlementOrder();
        order.create(settlementNo, orderNo, merchantId, merchantName, orderAmount, commissionRate);
        repository.save(order);
        eventPublisher.publishSettlementCreated(order.getId(), settlementNo, merchantId, orderAmount);
        return order.getId();
    }

    public void confirm(Long settlementId) {
        SettlementOrder order = repository.findById(settlementId)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found"));
        order.confirm();
        repository.update(order);
    }

    public void process(Long settlementId) {
        SettlementOrder order = repository.findById(settlementId)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found"));
        order.process();
        repository.update(order);
    }

    public void complete(Long settlementId) {
        SettlementOrder order = repository.findById(settlementId)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found"));
        order.complete();
        repository.update(order);
        eventPublisher.publishSettlementCompleted(settlementId, order.getSettlementNo());
    }

    public void fail(Long settlementId, String reason) {
        SettlementOrder order = repository.findById(settlementId)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found"));
        order.fail(reason);
        repository.update(order);
        eventPublisher.publishSettlementFailed(settlementId, order.getSettlementNo(), reason);
    }

    public void cancel(Long settlementId) {
        SettlementOrder order = repository.findById(settlementId)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found"));
        order.cancel();
        repository.update(order);
    }

    public void applyRefund(Long settlementId, BigDecimal amount) {
        SettlementOrder order = repository.findById(settlementId)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found"));
        order.applyRefund(amount);
        repository.update(order);
    }
}