package com.metawebthree.settlement.infrastructure.event;

import com.metawebthree.event.BaseEvent;
import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

/**
 * Settlement domain event publisher.
 */
@Component
public class SettlementEventPublisher {

    private final EventPublisher eventPublisher;

    public SettlementEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    /**
     * Publish settlement created event.
     */
    public void publishSettlementCreated(Long settlementId, String settlementNo, 
                                          Long merchantId, BigDecimal amount) {
        SettlementCreatedEvent event = SettlementCreatedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.SETTLEMENT_CREATED)
                .timestamp(Instant.now())
                .correlationId(settlementNo)
                .sourceService("settlement-service")
                .settlementId(settlementId.toString())
                .settlementNo(settlementNo)
                .merchantId(merchantId.toString())
                .amount(amount)
                .build();
        eventPublisher.publish(event, settlementNo);
    }

    /**
     * Publish settlement completed event.
     */
    public void publishSettlementCompleted(Long settlementId, String settlementNo) {
        SettlementCompletedEvent event = SettlementCompletedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.SETTLEMENT_COMPLETED)
                .timestamp(Instant.now())
                .correlationId(settlementNo)
                .sourceService("settlement-service")
                .settlementId(settlementId.toString())
                .settlementNo(settlementNo)
                .build();
        eventPublisher.publish(event, settlementNo);
    }

    /**
     * Publish settlement failed event.
     */
    public void publishSettlementFailed(Long settlementId, String settlementNo, String reason) {
        SettlementFailedEvent event = SettlementFailedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.SETTLEMENT_FAILED)
                .timestamp(Instant.now())
                .correlationId(settlementNo)
                .sourceService("settlement-service")
                .settlementId(settlementId.toString())
                .settlementNo(settlementNo)
                .reason(reason)
                .build();
        eventPublisher.publish(event, settlementNo);
    }

    /**
     * Publish reconciliation completed event.
     */
    public void publishReconciliationCompleted(String reconciliationNo, String period, String result) {
        ReconciliationCompletedEvent event = ReconciliationCompletedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.RECONCILIATION_COMPLETED)
                .timestamp(Instant.now())
                .correlationId(reconciliationNo)
                .sourceService("settlement-service")
                .reconciliationNo(reconciliationNo)
                .period(period)
                .result(result)
                .build();
        eventPublisher.publish(event, reconciliationNo);
    }

    // Event classes
    public static class SettlementCreatedEvent extends BaseEvent {
        private String settlementId;
        private String settlementNo;
        private String merchantId;
        private BigDecimal amount;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final SettlementCreatedEvent e = new SettlementCreatedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder settlementId(String v) { e.settlementId = v; return this; }
            public Builder settlementNo(String v) { e.settlementNo = v; return this; }
            public Builder merchantId(String v) { e.merchantId = v; return this; }
            public Builder amount(BigDecimal v) { e.amount = v; return this; }
            public SettlementCreatedEvent build() { return e; }
        }
        public String getSettlementId() { return settlementId; }
        public String getSettlementNo() { return settlementNo; }
        public String getMerchantId() { return merchantId; }
        public BigDecimal getAmount() { return amount; }
    }

    public static class SettlementCompletedEvent extends BaseEvent {
        private String settlementId;
        private String settlementNo;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final SettlementCompletedEvent e = new SettlementCompletedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder settlementId(String v) { e.settlementId = v; return this; }
            public Builder settlementNo(String v) { e.settlementNo = v; return this; }
            public SettlementCompletedEvent build() { return e; }
        }
        public String getSettlementId() { return settlementId; }
        public String getSettlementNo() { return settlementNo; }
    }

    public static class SettlementFailedEvent extends BaseEvent {
        private String settlementId;
        private String settlementNo;
        private String reason;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final SettlementFailedEvent e = new SettlementFailedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder settlementId(String v) { e.settlementId = v; return this; }
            public Builder settlementNo(String v) { e.settlementNo = v; return this; }
            public Builder reason(String v) { e.reason = v; return this; }
            public SettlementFailedEvent build() { return e; }
        }
        public String getSettlementId() { return settlementId; }
        public String getSettlementNo() { return settlementNo; }
        public String getReason() { return reason; }
    }

    public static class ReconciliationCompletedEvent extends BaseEvent {
        private String reconciliationNo;
        private String period;
        private String result;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final ReconciliationCompletedEvent e = new ReconciliationCompletedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder reconciliationNo(String v) { e.reconciliationNo = v; return this; }
            public Builder period(String v) { e.period = v; return this; }
            public Builder result(String v) { e.result = v; return this; }
            public ReconciliationCompletedEvent build() { return e; }
        }
        public String getReconciliationNo() { return reconciliationNo; }
        public String getPeriod() { return period; }
        public String getResult() { return result; }
    }
}