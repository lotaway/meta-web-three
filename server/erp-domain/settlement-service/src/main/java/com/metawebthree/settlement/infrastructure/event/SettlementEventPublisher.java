package com.metawebthree.settlement.infrastructure.event;

import com.metawebthree.event.BaseEvent;
import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.math.BigDecimal;

@Component
public class SettlementEventPublisher {

    private final EventPublisher eventPublisher;

    public SettlementEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishSettlementCreated(Long settlementId, String settlementNo, 
                                          Long merchantId, BigDecimal amount) {
        SettlementCreatedEvent event = SettlementCreatedEvent.builder()
                .eventType(EventType.SETTLEMENT_CREATED)
                .correlationId(settlementNo)
                .sourceService("settlement-service")
                .settlementId(settlementId.toString())
                .settlementNo(settlementNo)
                .merchantId(merchantId.toString())
                .amount(amount)
                .build();
        eventPublisher.publish(event, settlementNo);
    }

    public void publishSettlementCompleted(Long settlementId, String settlementNo) {
        SettlementCompletedEvent event = SettlementCompletedEvent.builder()
                .eventType(EventType.SETTLEMENT_COMPLETED)
                .correlationId(settlementNo)
                .sourceService("settlement-service")
                .settlementId(settlementId.toString())
                .settlementNo(settlementNo)
                .build();
        eventPublisher.publish(event, settlementNo);
    }

    public void publishSettlementFailed(Long settlementId, String settlementNo, String reason) {
        SettlementFailedEvent event = SettlementFailedEvent.builder()
                .eventType(EventType.SETTLEMENT_FAILED)
                .correlationId(settlementNo)
                .sourceService("settlement-service")
                .settlementId(settlementId.toString())
                .settlementNo(settlementNo)
                .reason(reason)
                .build();
        eventPublisher.publish(event, settlementNo);
    }

    public void publishReconciliationCompleted(String reconciliationNo, String period, String result) {
        ReconciliationCompletedEvent event = ReconciliationCompletedEvent.builder()
                .eventType(EventType.RECONCILIATION_COMPLETED)
                .correlationId(reconciliationNo)
                .sourceService("settlement-service")
                .reconciliationNo(reconciliationNo)
                .period(period)
                .result(result)
                .build();
        eventPublisher.publish(event, reconciliationNo);
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class SettlementCreatedEvent extends BaseEvent {
        private String settlementId;
        private String settlementNo;
        private String merchantId;
        private BigDecimal amount;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class SettlementCompletedEvent extends BaseEvent {
        private String settlementId;
        private String settlementNo;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class SettlementFailedEvent extends BaseEvent {
        private String settlementId;
        private String settlementNo;
        private String reason;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class ReconciliationCompletedEvent extends BaseEvent {
        private String reconciliationNo;
        private String period;
        private String result;
    }
}