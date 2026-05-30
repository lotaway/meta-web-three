package com.metawebthree.finance.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Account {
    private Long id;
    private String accountNo;
    private String accountName;
    private AccountType type;
    private BigDecimal balance;
    private AccountStatus status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum AccountType {
        CASH, BANK, VIRTUAL, CREDIT
    }

    public enum AccountStatus {
        ACTIVE, FROZEN, CLOSED
    }

    public void create(String accountNo, String accountName, AccountType type) {
        this.accountNo = accountNo;
        this.accountName = accountName;
        this.type = type;
        this.balance = BigDecimal.ZERO;
        this.status = AccountStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void freeze() {
        if (status != AccountStatus.ACTIVE) {
            return;
        }
        status = AccountStatus.FROZEN;
        updatedAt = LocalDateTime.now();
    }

    public void unfreeze() {
        if (status != AccountStatus.FROZEN) {
            return;
        }
        status = AccountStatus.ACTIVE;
        updatedAt = LocalDateTime.now();
    }

    public void close() {
        if (balance.compareTo(BigDecimal.ZERO) != 0) {
            throw new IllegalStateException("Cannot close account with balance");
        }
        status = AccountStatus.CLOSED;
        updatedAt = LocalDateTime.now();
    }

    public void credit(BigDecimal amount) {
        validateActive();
        validateAmount(amount);
        balance = balance.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public void debit(BigDecimal amount) {
        validateActive();
        validateAmount(amount);
        if (balance.compareTo(amount) < 0) {
            throw new IllegalStateException("Insufficient balance");
        }
        balance = balance.subtract(amount);
        updatedAt = LocalDateTime.now();
    }

    private void validateActive() {
        if (status != AccountStatus.ACTIVE) {
            throw new IllegalStateException("Account is not active");
        }
    }

    private void validateAmount(BigDecimal amount) {
        if (amount == null || amount.compareTo(BigDecimal.ZERO) <= 0) {
            throw new IllegalArgumentException("Invalid amount");
        }
    }
}