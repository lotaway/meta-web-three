package com.metawebthree.finance.infrastructure.event;

import com.metawebthree.event.BaseEvent;
import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

/**
 * Finance domain event publisher.
 */
@Component
public class FinanceEventPublisher {

    private final EventPublisher eventPublisher;

    public FinanceEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    /**
     * Publish account created event.
     */
    public void publishAccountCreated(Long accountId, String accountNo, String accountName, String type) {
        AccountCreatedEvent event = AccountCreatedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.ACCOUNT_CREATED)
                .timestamp(Instant.now())
                .correlationId(accountNo)
                .sourceService("finance-service")
                .accountId(accountId.toString())
                .accountNo(accountNo)
                .accountName(accountName)
                .accountType(type)
                .build();
        eventPublisher.publish(event, accountNo);
    }

    /**
     * Publish account balance changed event.
     */
    public void publishAccountBalanceChanged(Long accountId, String accountNo, 
                                               BigDecimal beforeAmount, BigDecimal afterAmount, 
                                               String changeType) {
        AccountBalanceChangedEvent event = AccountBalanceChangedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.ACCOUNT_BALANCE_CHANGED)
                .timestamp(Instant.now())
                .correlationId(accountNo)
                .sourceService("finance-service")
                .accountId(accountId.toString())
                .accountNo(accountNo)
                .beforeAmount(beforeAmount)
                .afterAmount(afterAmount)
                .changeType(changeType)
                .build();
        eventPublisher.publish(event, accountNo);
    }

    /**
     * Publish voucher created event.
     */
    public void publishVoucherCreated(Long voucherId, String voucherNo, String status) {
        VoucherCreatedEvent event = VoucherCreatedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.VOUCHER_CREATED)
                .timestamp(Instant.now())
                .correlationId(voucherNo)
                .sourceService("finance-service")
                .voucherId(voucherId.toString())
                .voucherNo(voucherNo)
                .status(status)
                .build();
        eventPublisher.publish(event, voucherNo);
    }

    /**
     * Publish voucher posted event.
     */
    public void publishVoucherPosted(Long voucherId, String voucherNo) {
        VoucherPostedEvent event = VoucherPostedEvent.builder()
                .eventId(UUID.randomUUID().toString())
                .eventType(EventType.VOUCHER_POSTED)
                .timestamp(Instant.now())
                .correlationId(voucherNo)
                .sourceService("finance-service")
                .voucherId(voucherId.toString())
                .voucherNo(voucherNo)
                .build();
        eventPublisher.publish(event, voucherNo);
    }

    // Event classes
    public static class AccountCreatedEvent extends BaseEvent {
        private String accountId;
        private String accountNo;
        private String accountName;
        private String accountType;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final AccountCreatedEvent e = new AccountCreatedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder accountId(String v) { e.accountId = v; return this; }
            public Builder accountNo(String v) { e.accountNo = v; return this; }
            public Builder accountName(String v) { e.accountName = v; return this; }
            public Builder accountType(String v) { e.accountType = v; return this; }
            public AccountCreatedEvent build() { return e; }
        }
        public String getAccountId() { return accountId; }
        public String getAccountNo() { return accountNo; }
        public String getAccountName() { return accountName; }
        public String getAccountType() { return accountType; }
    }

    public static class AccountBalanceChangedEvent extends BaseEvent {
        private String accountId;
        private String accountNo;
        private BigDecimal beforeAmount;
        private BigDecimal afterAmount;
        private String changeType;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final AccountBalanceChangedEvent e = new AccountBalanceChangedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder accountId(String v) { e.accountId = v; return this; }
            public Builder accountNo(String v) { e.accountNo = v; return this; }
            public Builder beforeAmount(BigDecimal v) { e.beforeAmount = v; return this; }
            public Builder afterAmount(BigDecimal v) { e.afterAmount = v; return this; }
            public Builder changeType(String v) { e.changeType = v; return this; }
            public AccountBalanceChangedEvent build() { return e; }
        }
        public String getAccountId() { return accountId; }
        public String getAccountNo() { return accountNo; }
        public BigDecimal getBeforeAmount() { return beforeAmount; }
        public BigDecimal getAfterAmount() { return afterAmount; }
        public String getChangeType() { return changeType; }
    }

    public static class VoucherCreatedEvent extends BaseEvent {
        private String voucherId;
        private String voucherNo;
        private String status;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final VoucherCreatedEvent e = new VoucherCreatedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder voucherId(String v) { e.voucherId = v; return this; }
            public Builder voucherNo(String v) { e.voucherNo = v; return this; }
            public Builder status(String v) { e.status = v; return this; }
            public VoucherCreatedEvent build() { return e; }
        }
        public String getVoucherId() { return voucherId; }
        public String getVoucherNo() { return voucherNo; }
        public String getStatus() { return status; }
    }

    public static class VoucherPostedEvent extends BaseEvent {
        private String voucherId;
        private String voucherNo;

        public static Builder builder() { return new Builder(); }
        public static class Builder {
            private final VoucherPostedEvent e = new VoucherPostedEvent();
            public Builder eventId(String v) { e.eventId = v; return this; }
            public Builder eventType(EventType v) { e.eventType = v; return this; }
            public Builder timestamp(Instant v) { e.timestamp = v; return this; }
            public Builder correlationId(String v) { e.correlationId = v; return this; }
            public Builder sourceService(String v) { e.sourceService = v; return this; }
            public Builder voucherId(String v) { e.voucherId = v; return this; }
            public Builder voucherNo(String v) { e.voucherNo = v; return this; }
            public VoucherPostedEvent build() { return e; }
        }
        public String getVoucherId() { return voucherId; }
        public String getVoucherNo() { return voucherNo; }
    }
}