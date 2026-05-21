package com.metawebthree.commission.application;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import org.springframework.transaction.annotation.Transactional;

import com.metawebthree.commission.domain.CommissionAccount;
import com.metawebthree.commission.domain.CommissionConfig;
import com.metawebthree.commission.domain.CommissionRecord;
import com.metawebthree.commission.domain.CommissionRecordStatus;
import com.metawebthree.commission.domain.CommissionRelation;
import com.metawebthree.commission.domain.ports.CommissionAccountStore;
import com.metawebthree.commission.domain.ports.CommissionRecordStore;
import com.metawebthree.commission.domain.ports.CommissionRelationStore;

public class CommissionCommandService {
    private final CommissionRelationStore relationStore;
    private final CommissionRecordStore recordStore;
    private final CommissionAccountStore accountStore;
    private final CommissionConfigProvider configProvider;
    private final TimeProvider timeProvider;

    public CommissionCommandService(CommissionRelationStore relationStore,
            CommissionRecordStore recordStore,
            CommissionAccountStore accountStore,
            CommissionConfigProvider configProvider,
            TimeProvider timeProvider) {
        this.relationStore = relationStore;
        this.recordStore = recordStore;
        this.accountStore = accountStore;
        this.configProvider = configProvider;
        this.timeProvider = timeProvider;
    }

    @Transactional
    public void bindRelation(Long userId, Long parentUserId) {
        validateRelationInput(userId, parentUserId);
        CommissionRelation existing = relationStore.findByUserId(userId);
        ensureRelationNotChanged(existing, parentUserId);
        if (existing != null) {
            return;
        }
        CommissionRelation record = buildRelation(userId, parentUserId);
        relationStore.save(record);
    }

    @Transactional
    public void calculateForOrder(Long orderId, Long userId, BigDecimal payAmount, LocalDateTime availableAt) {
        validateOrderInput(orderId, userId, payAmount, availableAt);
        ensureOrderNotCalculated(orderId);
        CommissionConfig config = configProvider.getConfig();
        List<Long> uplines = loadUplines(userId, config.getMaxLevels());
        createCommissionRecords(orderId, userId, payAmount, availableAt, uplines, config);
    }

    @Transactional
    public void settleBefore(LocalDateTime executeBefore) {
        validateTime(executeBefore);
        List<CommissionRecord> records = recordStore.findPendingBefore(executeBefore);
        for (CommissionRecord record : records) {
            settleRecord(record);
        }
    }

    @Transactional
    public void cancelByOrderId(Long orderId) {
        validateOrderId(orderId);
        List<CommissionRecord> records = recordStore.findActiveByOrderId(orderId);
        for (CommissionRecord record : records) {
            cancelRecord(record);
        }
    }

    private void validateRelationInput(Long userId, Long parentUserId) {
        if (userId == null || parentUserId == null || parentUserId <= 0) {
            throw new IllegalArgumentException("invalid relation input");
        }
        if (userId.equals(parentUserId)) {
            throw new IllegalArgumentException("userId equals parentUserId");
        }
    }

    private void validateOrderInput(Long orderId, Long userId, BigDecimal payAmount, LocalDateTime availableAt) {
        if (orderId == null || userId == null || availableAt == null) {
            throw new IllegalArgumentException("invalid order input");
        }
        if (payAmount == null || payAmount.signum() <= 0) {
            throw new IllegalArgumentException("invalid pay amount");
        }
    }

    private void validateTime(LocalDateTime time) {
        if (time == null) {
            throw new IllegalArgumentException("invalid time");
        }
    }

    private void validateOrderId(Long orderId) {
        if (orderId == null) {
            throw new IllegalArgumentException("invalid order id");
        }
    }

    private void ensureRelationNotChanged(CommissionRelation existing, Long parentUserId) {
        if (existing == null) {
            return;
        }
        if (!existing.getParentUserId().equals(parentUserId)) {
            throw new IllegalStateException("relation already exists");
        }
    }

    private void ensureOrderNotCalculated(Long orderId) {
        if (recordStore.countActiveByOrderId(orderId) > 0) {
            throw new IllegalStateException("commission already calculated");
        }
    }

    private CommissionRelation buildRelation(Long userId, Long parentUserId) {
        CommissionRelation parent = relationStore.findByUserId(parentUserId);
        int parentDepth = parent == null || parent.getDepth() == null ? 1 : parent.getDepth();
        CommissionRelation record = new CommissionRelation();
        record.setUserId(userId);
        record.setParentUserId(parentUserId);
        record.setDepth(parentDepth + 1);
        LocalDateTime now = timeProvider.now();
        record.setCreatedAt(now);
        record.setUpdatedAt(now);
        return record;
    }

    private List<Long> loadUplines(Long userId, int maxLevels) {
        List<Long> uplines = new ArrayList<>();
        Long current = userId;
        int depth = 0;
        while (depth < maxLevels) {
        CommissionRelation relation = relationStore.findByUserId(current);
            if (relation == null || relation.getParentUserId() == null) {
                break;
            }
            Long parentId = relation.getParentUserId();
            uplines.add(parentId);
            current = parentId;
            depth++;
        }
        return uplines;
    }

    private void createCommissionRecords(Long orderId, Long userId, BigDecimal payAmount,
            LocalDateTime availableAt, List<Long> uplines, CommissionConfig config) {
        if (uplines.isEmpty()) {
            return;
        }
        BigDecimal commissionTotal = payAmount.multiply(config.getBuyRate());
        List<BigDecimal> rates = config.getLevelRates();
        for (int i = 0; i < uplines.size(); i++) {
            if (i >= rates.size()) {
                return;
            }
            createCommissionRecord(orderId, userId, uplines.get(i), i, commissionTotal, rates.get(i), availableAt);
        }
    }

    private void createCommissionRecord(Long orderId, Long userId, Long uplineId, int index,
            BigDecimal commissionTotal, BigDecimal rate, LocalDateTime availableAt) {
        if (rate == null || rate.signum() <= 0) {
            return;
        }
        BigDecimal amount = commissionTotal.multiply(rate);
        if (amount.signum() <= 0) {
            return;
        }
        CommissionRecord record = new CommissionRecord();
        record.setOrderId(orderId);
        record.setUserId(uplineId);
        record.setFromUserId(userId);
        record.setLevel(index + 1);
        record.setAmount(amount);
        record.setStatus(CommissionRecordStatus.PENDING.name());
        record.setAvailableAt(availableAt);
        LocalDateTime now = timeProvider.now();
        record.setCreatedAt(now);
        record.setUpdatedAt(now);
        recordStore.save(record);
        increaseFrozen(uplineId, amount, now);
    }

    private void settleRecord(CommissionRecord record) {
        LocalDateTime now = timeProvider.now();
        boolean updated = recordStore.updateStatus(record.getId(),
                CommissionRecordStatus.PENDING.name(),
                CommissionRecordStatus.AVAILABLE.name(),
                now);
        if (updated) {
            moveFrozenToAvailable(record.getUserId(), record.getAmount(), now);
        }
    }

    private void cancelRecord(CommissionRecord record) {
        LocalDateTime now = timeProvider.now();
        recordStore.updateStatus(record.getId(), CommissionRecordStatus.CANCELED.name(), now);
        adjustAccountForCancel(record, now);
    }

    private void adjustAccountForCancel(CommissionRecord record, LocalDateTime now) {
        String status = record.getStatus();
        if (CommissionRecordStatus.PENDING.name().equals(status)) {
            decreaseFrozen(record.getUserId(), record.getAmount(), now);
            return;
        }
        if (CommissionRecordStatus.AVAILABLE.name().equals(status)) {
            decreaseAvailable(record.getUserId(), record.getAmount(), now);
        }
    }

    private void increaseFrozen(Long userId, BigDecimal amount, LocalDateTime now) {
        CommissionAccount account = ensureAccount(userId, now);
        CommissionAccount updated = cloneAccount(account);
        updated.setTotalAmount(safe(account.getTotalAmount()).add(amount));
        updated.setFrozenAmount(safe(account.getFrozenAmount()).add(amount));
        accountStore.updateBalances(account.getId(), updated, now);
    }

    private void moveFrozenToAvailable(Long userId, BigDecimal amount, LocalDateTime now) {
        CommissionAccount account = ensureAccount(userId, now);
        CommissionAccount updated = cloneAccount(account);
        updated.setFrozenAmount(safe(account.getFrozenAmount()).subtract(amount).max(BigDecimal.ZERO));
        updated.setAvailableAmount(safe(account.getAvailableAmount()).add(amount));
        accountStore.updateBalances(account.getId(), updated, now);
    }

    private void decreaseFrozen(Long userId, BigDecimal amount, LocalDateTime now) {
        CommissionAccount account = ensureAccount(userId, now);
        CommissionAccount updated = cloneAccount(account);
        updated.setFrozenAmount(safe(account.getFrozenAmount()).subtract(amount).max(BigDecimal.ZERO));
        updated.setTotalAmount(safe(account.getTotalAmount()).subtract(amount).max(BigDecimal.ZERO));
        accountStore.updateBalances(account.getId(), updated, now);
    }

    private void decreaseAvailable(Long userId, BigDecimal amount, LocalDateTime now) {
        CommissionAccount account = ensureAccount(userId, now);
        CommissionAccount updated = cloneAccount(account);
        updated.setAvailableAmount(safe(account.getAvailableAmount()).subtract(amount).max(BigDecimal.ZERO));
        updated.setTotalAmount(safe(account.getTotalAmount()).subtract(amount).max(BigDecimal.ZERO));
        accountStore.updateBalances(account.getId(), updated, now);
    }

    private CommissionAccount ensureAccount(Long userId, LocalDateTime now) {
        CommissionAccount account = accountStore.findByUserId(userId);
        if (account != null) {
            return account;
        }
        CommissionAccount record = new CommissionAccount();
        record.setUserId(userId);
        record.setTotalAmount(BigDecimal.ZERO);
        record.setAvailableAmount(BigDecimal.ZERO);
        record.setFrozenAmount(BigDecimal.ZERO);
        record.setCreatedAt(now);
        record.setUpdatedAt(now);
        accountStore.save(record);
        return accountStore.findByUserId(userId);
    }

    private CommissionAccount cloneAccount(CommissionAccount account) {
        CommissionAccount record = new CommissionAccount();
        record.setId(account.getId());
        record.setUserId(account.getUserId());
        record.setTotalAmount(account.getTotalAmount());
        record.setAvailableAmount(account.getAvailableAmount());
        record.setFrozenAmount(account.getFrozenAmount());
        record.setCreatedAt(account.getCreatedAt());
        record.setUpdatedAt(account.getUpdatedAt());
        return record;
    }

    private BigDecimal safe(BigDecimal value) {
        return value == null ? BigDecimal.ZERO : value;
    }
}
