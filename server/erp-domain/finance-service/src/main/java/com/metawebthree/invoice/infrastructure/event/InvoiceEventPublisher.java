package com.metawebthree.invoice.infrastructure.event;

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
public class InvoiceEventPublisher {

    private final EventPublisher eventPublisher;

    public InvoiceEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishInvoiceCreated(Long invoiceId, String invoiceNo, Long customerId, 
                                        BigDecimal amount) {
        InvoiceCreatedEvent event = InvoiceCreatedEvent.builder()
                .eventType(EventType.INVOICE_CREATED)
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .customerId(customerId.toString())
                .amount(amount)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    public void publishInvoiceIssued(Long invoiceId, String invoiceNo) {
        InvoiceIssuedEvent event = InvoiceIssuedEvent.builder()
                .eventType(EventType.INVOICE_ISSUED)
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    public void publishInvoiceVoided(Long invoiceId, String invoiceNo, String reason) {
        InvoiceVoidedEvent event = InvoiceVoidedEvent.builder()
                .eventType(EventType.INVOICE_VOIDED)
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .reason(reason)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    public void publishInvoiceRedFlushed(Long invoiceId, String invoiceNo, String reason) {
        InvoiceRedFlushedEvent event = InvoiceRedFlushedEvent.builder()
                .eventType(EventType.INVOICE_RED_FLUSHED)
                .correlationId(invoiceNo)
                .sourceService("invoice-service")
                .invoiceId(invoiceId.toString())
                .invoiceNo(invoiceNo)
                .reason(reason)
                .build();
        eventPublisher.publish(event, invoiceNo);
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class InvoiceCreatedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
        private String customerId;
        private BigDecimal amount;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class InvoiceIssuedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class InvoiceVoidedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
        private String reason;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @SuperBuilder
    public static class InvoiceRedFlushedEvent extends BaseEvent {
        private String invoiceId;
        private String invoiceNo;
        private String reason;
    }
}