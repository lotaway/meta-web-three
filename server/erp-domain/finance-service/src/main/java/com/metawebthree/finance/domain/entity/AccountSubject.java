package com.metawebthree.finance.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class AccountSubject {
    private Long id;
    private String subjectCode;
    private String subjectName;
    private SubjectDirection direction;
    private Long parentId;
    private Integer level;
    private SubjectStatus status;
    private BigDecimal balance;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum SubjectDirection {
        DEBIT, CREDIT
    }

    public enum SubjectStatus {
        ACTIVE, DISABLED
    }

    public void create(String subjectCode, String subjectName, SubjectDirection direction, Long parentId) {
        this.subjectCode = subjectCode;
        this.subjectName = subjectName;
        this.direction = direction;
        this.parentId = parentId;
        this.level = parentId == null ? 1 : 2;
        this.status = SubjectStatus.ACTIVE;
        this.balance = BigDecimal.ZERO;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void disable() {
        if (status != SubjectStatus.ACTIVE) {
            return;
        }
        status = SubjectStatus.DISABLED;
        updatedAt = LocalDateTime.now();
    }

    public void enable() {
        if (status != SubjectStatus.DISABLED) {
            return;
        }
        status = SubjectStatus.ACTIVE;
        updatedAt = LocalDateTime.now();
    }

    public void updateBalance(BigDecimal amount) {
        balance = balance.add(amount);
        updatedAt = LocalDateTime.now();
    }

    public boolean isDebitDirection() {
        return direction == SubjectDirection.DEBIT;
    }

    public Long getId() { return id; }
    public String getSubjectCode() { return subjectCode; }
    public String getSubjectName() { return subjectName; }
    public SubjectDirection getDirection() { return direction; }
    public Long getParentId() { return parentId; }
    public Integer getLevel() { return level; }
    public SubjectStatus getStatus() { return status; }
    public BigDecimal getBalance() { return balance; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setId(Long id) { this.id = id; }
    public void setBalance(BigDecimal balance) { this.balance = balance; }
}