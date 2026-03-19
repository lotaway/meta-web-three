package com.metawebthree.commission.infrastructure.persistence.repository;

import java.time.LocalDateTime;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.metawebthree.commission.domain.CommissionAccount;
import com.metawebthree.commission.domain.ports.CommissionAccountStore;
import com.metawebthree.commission.infrastructure.persistence.mapper.CommissionAccountMapper;
import com.metawebthree.commission.infrastructure.persistence.model.CommissionAccountRecord;

@Repository
public class MybatisCommissionAccountStore implements CommissionAccountStore {
    private final CommissionAccountMapper mapper;

    public MybatisCommissionAccountStore(CommissionAccountMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public CommissionAccount findByUserId(Long userId) {
        CommissionAccountRecord record = mapper.selectOne(new LambdaQueryWrapper<CommissionAccountRecord>()
                .eq(CommissionAccountRecord::getUserId, userId));
        return record == null ? null : toDomain(record);
    }

    @Override
    public void save(CommissionAccount account) {
        mapper.insert(toRecord(account));
    }

    @Override
    public void updateBalances(Long id, CommissionAccount updated, LocalDateTime updatedAt) {
        LambdaUpdateWrapper<CommissionAccountRecord> update = new LambdaUpdateWrapper<CommissionAccountRecord>()
                .eq(CommissionAccountRecord::getId, id)
                .set(CommissionAccountRecord::getTotalAmount, updated.getTotalAmount())
                .set(CommissionAccountRecord::getAvailableAmount, updated.getAvailableAmount())
                .set(CommissionAccountRecord::getFrozenAmount, updated.getFrozenAmount())
                .set(CommissionAccountRecord::getUpdatedAt, updatedAt);
        mapper.update(null, update);
    }

    private CommissionAccount toDomain(CommissionAccountRecord record) {
        CommissionAccount account = new CommissionAccount();
        account.setId(record.getId());
        account.setUserId(record.getUserId());
        account.setTotalAmount(record.getTotalAmount());
        account.setAvailableAmount(record.getAvailableAmount());
        account.setFrozenAmount(record.getFrozenAmount());
        account.setCreatedAt(record.getCreatedAt());
        account.setUpdatedAt(record.getUpdatedAt());
        return account;
    }

    private CommissionAccountRecord toRecord(CommissionAccount account) {
        CommissionAccountRecord record = new CommissionAccountRecord();
        record.setId(account.getId());
        record.setUserId(account.getUserId());
        record.setTotalAmount(account.getTotalAmount());
        record.setAvailableAmount(account.getAvailableAmount());
        record.setFrozenAmount(account.getFrozenAmount());
        record.setCreatedAt(account.getCreatedAt());
        record.setUpdatedAt(account.getUpdatedAt());
        return record;
    }
}
