package com.metawebthree.promotion.infrastructure.persistence.repository;

import org.springframework.stereotype.Repository;

import com.metawebthree.promotion.domain.model.CouponBatch;
import com.metawebthree.promotion.domain.ports.CouponBatchRepository;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CouponBatchMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CouponBatchRecord;

@Repository
public class MybatisCouponBatchRepository implements CouponBatchRepository {
    private final CouponBatchMapper mapper;

    public MybatisCouponBatchRepository(CouponBatchMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public void save(CouponBatch batch) {
        mapper.insert(toRecord(batch));
    }

    @Override
    public CouponBatch findById(String id) {
        CouponBatchRecord record = mapper.selectById(id);
        if (record == null) {
            return null;
        }
        return toDomain(record);
    }

    private CouponBatchRecord toRecord(CouponBatch batch) {
        CouponBatchRecord record = new CouponBatchRecord();
        record.setId(batch.getId());
        record.setCouponTypeId(batch.getCouponTypeId());
        record.setTotalCount(batch.getTotalCount());
        record.setCreatedAt(batch.getCreatedAt());
        return record;
    }

    private CouponBatch toDomain(CouponBatchRecord record) {
        CouponBatch batch = new CouponBatch();
        batch.setId(record.getId());
        batch.setCouponTypeId(record.getCouponTypeId());
        batch.setTotalCount(record.getTotalCount());
        batch.setCreatedAt(record.getCreatedAt());
        return batch;
    }
}
