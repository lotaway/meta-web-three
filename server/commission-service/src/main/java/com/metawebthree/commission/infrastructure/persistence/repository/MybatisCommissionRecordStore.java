package com.metawebthree.commission.infrastructure.persistence.repository;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.metawebthree.commission.domain.CommissionRecord;
import com.metawebthree.commission.domain.CommissionRecordStatus;
import com.metawebthree.commission.domain.ports.CommissionRecordStore;
import com.metawebthree.commission.infrastructure.persistence.mapper.CommissionRecordMapper;
import com.metawebthree.commission.infrastructure.persistence.model.CommissionRecordRecord;

@Repository
public class MybatisCommissionRecordStore implements CommissionRecordStore {
    private final CommissionRecordMapper mapper;

    public MybatisCommissionRecordStore(CommissionRecordMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public long countActiveByOrderId(Long orderId) {
        return mapper.selectCount(new LambdaQueryWrapper<CommissionRecordRecord>()
                .eq(CommissionRecordRecord::getOrderId, orderId)
                .ne(CommissionRecordRecord::getStatus, CommissionRecordStatus.CANCELED.name()));
    }

    @Override
    public void save(CommissionRecord record) {
        mapper.insert(toRecord(record));
    }

    @Override
    public List<CommissionRecord> findPendingBefore(LocalDateTime executeBefore) {
        List<CommissionRecordRecord> records = mapper.selectList(new LambdaQueryWrapper<CommissionRecordRecord>()
                .eq(CommissionRecordRecord::getStatus, CommissionRecordStatus.PENDING.name())
                .le(CommissionRecordRecord::getAvailableAt, executeBefore));
        return toDomain(records);
    }

    @Override
    public List<CommissionRecord> findActiveByOrderId(Long orderId) {
        List<CommissionRecordRecord> records = mapper.selectList(new LambdaQueryWrapper<CommissionRecordRecord>()
                .eq(CommissionRecordRecord::getOrderId, orderId)
                .ne(CommissionRecordRecord::getStatus, CommissionRecordStatus.CANCELED.name()));
        return toDomain(records);
    }

    @Override
    public boolean updateStatus(Long id, String fromStatus, String toStatus, LocalDateTime updatedAt) {
        LambdaUpdateWrapper<CommissionRecordRecord> update = new LambdaUpdateWrapper<CommissionRecordRecord>()
                .eq(CommissionRecordRecord::getId, id)
                .eq(CommissionRecordRecord::getStatus, fromStatus)
                .set(CommissionRecordRecord::getStatus, toStatus)
                .set(CommissionRecordRecord::getUpdatedAt, updatedAt);
        return mapper.update(null, update) == 1;
    }

    @Override
    public boolean updateStatus(Long id, String toStatus, LocalDateTime updatedAt) {
        LambdaUpdateWrapper<CommissionRecordRecord> update = new LambdaUpdateWrapper<CommissionRecordRecord>()
                .eq(CommissionRecordRecord::getId, id)
                .set(CommissionRecordRecord::getStatus, toStatus)
                .set(CommissionRecordRecord::getUpdatedAt, updatedAt);
        return mapper.update(null, update) == 1;
    }

    @Override
    public List<CommissionRecord> findByUserId(Long userId, String status, int page, int size) {
        LambdaQueryWrapper<CommissionRecordRecord> query = new LambdaQueryWrapper<CommissionRecordRecord>()
                .eq(CommissionRecordRecord::getUserId, userId)
                .orderByDesc(CommissionRecordRecord::getCreatedAt)
                .last("limit " + size + " offset " + Math.max(0, (page - 1) * size));
        if (status != null && !status.isBlank()) {
            query.eq(CommissionRecordRecord::getStatus, status);
        }
        return toDomain(mapper.selectList(query));
    }

    private List<CommissionRecord> toDomain(List<CommissionRecordRecord> records) {
        List<CommissionRecord> result = new java.util.ArrayList<>();
        for (CommissionRecordRecord record : records) {
            result.add(toDomain(record));
        }
        return result;
    }

    private CommissionRecord toDomain(CommissionRecordRecord record) {
        CommissionRecord domain = new CommissionRecord();
        domain.setId(record.getId());
        domain.setOrderId(record.getOrderId());
        domain.setUserId(record.getUserId());
        domain.setFromUserId(record.getFromUserId());
        domain.setLevel(record.getLevel());
        domain.setAmount(record.getAmount());
        domain.setStatus(record.getStatus());
        domain.setAvailableAt(record.getAvailableAt());
        domain.setCreatedAt(record.getCreatedAt());
        domain.setUpdatedAt(record.getUpdatedAt());
        return domain;
    }

    private CommissionRecordRecord toRecord(CommissionRecord record) {
        CommissionRecordRecord entity = new CommissionRecordRecord();
        entity.setId(record.getId());
        entity.setOrderId(record.getOrderId());
        entity.setUserId(record.getUserId());
        entity.setFromUserId(record.getFromUserId());
        entity.setLevel(record.getLevel());
        entity.setAmount(record.getAmount());
        entity.setStatus(record.getStatus());
        entity.setAvailableAt(record.getAvailableAt());
        entity.setCreatedAt(record.getCreatedAt());
        entity.setUpdatedAt(record.getUpdatedAt());
        return entity;
    }
}
