package com.metawebthree.invoice.infrastructure.event;

import com.metawebthree.event.BaseEvent;
import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

/**
 * Invoice domain event publisher.
 */
@Component
public class InvoiceEventPublisher {

    private final EventPublisher eventPublisher;

    public InvoiceEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    /**
     * Publish invoice created event.
     */
    public void publishInvoiceCreated(Long invoiceId, String invoiceNo, Long customerId, 
                                        BigDecimal amount) {
        InvoiceCreatedEvent event = InvoiceCreatedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.INVOICE_CREATED)
                .timestamp(Instant.now())
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .customerId(customerId.toString())
                .amount(amount)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    /**
     * Publish invoice issued event.
     */
    public void publishInvoiceIssued(Long invoiceId, String invoiceNo) {
        InvoiceIssuedEvent event = InvoiceIssuedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.INVOICE_ISSUED)
                .timestamp(Instant.now())
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    /**
     * Publish invoice voided event.
     */
    public void publishInvoiceVoided(Long invoiceId, String invoiceNo, String reason) {
        InvoiceVoidedEvent event = InvoiceVoidedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.INVOICE_VOIDED)
                .timestamp(Instant.now())
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .reason(reason)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    /**
     * Publish invoice red-flushed event.
     */
    public void publishInvoiceRedFlushed(Long invoiceId, String invoiceNo, String reason) {
        InvoiceRedFlushedEvent event = InvoiceRedFlushedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.INVOICE_RED_FLUSHED)
                .timestamp(Instant.now())
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .reason(reason)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    // Event classes
    public static class InvoiceCreatedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
        private String customerId;
        private BigDecimal amount;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final InvoiceCreatedEvent e = new InvoiceCreatedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder invoiceId(String v) { e.invoiceId = v; return this; }
            public Builder invoiceNo(String v) { e.invoiceNo = v; return this; }
            public Builder customerId(String v) { e.customerId = v; return this; }
            public Builder amount(BigDecimal v) { e.amount = v; return this; }
            public InvoiceCreatedEvent build() { return e; }
        }
        public String getInvoiceId() { return invoiceId; }
        public String getInvoiceNo() { return invoiceNo; }
        public String getCustomerId() { return customerId; }
        public BigDecimal getAmount() { return amount; }
    }

    public static class InvoiceIssuedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final InvoiceIssuedEvent e = new InvoiceIssuedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder invoiceId(String v) { e.invoiceId = v; return this; }
            public Builder invoiceNo(String v) { e.invoiceNo = v; return this; }
            public InvoiceIssuedEvent build() { return e; }
        }
        public String getInvoiceId() { return invoiceId; }
        public String getInvoiceNo() { return invoiceNo; }
    }

    public static class InvoiceVoidedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
        private String reason;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final InvoiceVoidedEvent e = new InvoiceVoidedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder invoiceId(String v) { e.invoiceId = v; return this; }
            public Builder invoiceNo(String v) { e.invoiceNo = v; return this; }
            public Builder reason(String v) { e.reason = v; return this; }
            public InvoiceVoidedEvent build() { return e; }
        }
        public String getInvoiceId() { return invoiceId; }
        public String getInvoiceNo() { return invoiceNo; }
        public String getReason() { return reason; }
    }

    public static class InvoiceRedFlushedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
        private String reason;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final InvoiceRedFlushedEvent e = new InvoiceRedFlushedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder invoiceId(String v) { e.invoiceId = v; return this; }
            public Builder invoiceNo(String v) { e.invoiceNo = v; return this; }
            public Builder reason(String v) { e.reason = v; return this; }
            public InvoiceRedFlushedEvent build() { return e; }
        }
        public String getInvoiceId() { return invoiceId; }
        public String getInvoiceNo() { return invoiceNo; }
        public String getReason() { return reason; }
    }
}