package com.metawebthree.promotion.infrastructure.persistence.repository;

import org.springframework.stereotype.Repository;

import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CouponTypeMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CouponTypeRecord;

@Repository
public class MybatisCouponTypeRepository implements CouponTypeRepository {
    private final CouponTypeMapper mapper;

    public MybatisCouponTypeRepository(CouponTypeMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public void save(CouponType couponType) {
        CouponTypeRecord record = toRecord(couponType);
        mapper.insert(record);
        couponType.setId(record.getId());
    }

    @Override
    public CouponType findById(Long id) {
        CouponTypeRecord record = mapper.selectById(id);
        if (record == null) {
            return null;
        }
        return toDomain(record);
    }

    private CouponTypeRecord toRecord(CouponType couponType) {
        CouponTypeRecord record = new CouponTypeRecord();
        copyRecordBasic(record, couponType);
        copyRecordTime(record, couponType);
        return record;
    }

    private CouponType toDomain(CouponTypeRecord record) {
        CouponType couponType = new CouponType();
        copyDomainBasic(couponType, record);
        copyDomainTime(couponType, record);
        return couponType;
    }

    private void copyRecordBasic(CouponTypeRecord record, CouponType couponType) {
        record.setId(couponType.getId());
        record.setName(couponType.getName());
        record.setDescription(couponType.getDescription());
        record.setImageUrl(couponType.getImageUrl());
        record.setMinimumOrderAmount(couponType.getMinimumOrderAmount());
        record.setDiscountAmount(couponType.getDiscountAmount());
        record.setIsEnabled(couponType.getIsEnabled());
        record.setCreateUserId(couponType.getCreateUserId());
        record.setTypeCode(couponType.getTypeCode());
    }

    private void copyRecordTime(CouponTypeRecord record, CouponType couponType) {
        record.setStartTime(couponType.getStartTime());
        record.setEndTime(couponType.getEndTime());
        record.setCreatedAt(couponType.getCreatedAt());
        record.setUpdatedAt(couponType.getUpdatedAt());
    }

    private void copyDomainBasic(CouponType couponType, CouponTypeRecord record) {
        couponType.setId(record.getId());
        couponType.setName(record.getName());
        couponType.setDescription(record.getDescription());
        couponType.setImageUrl(record.getImageUrl());
        couponType.setMinimumOrderAmount(record.getMinimumOrderAmount());
        couponType.setDiscountAmount(record.getDiscountAmount());
        couponType.setIsEnabled(record.getIsEnabled());
        couponType.setCreateUserId(record.getCreateUserId());
        couponType.setTypeCode(record.getTypeCode());
    }

    private void copyDomainTime(CouponType couponType, CouponTypeRecord record) {
        couponType.setStartTime(record.getStartTime());
        couponType.setEndTime(record.getEndTime());
        couponType.setCreatedAt(record.getCreatedAt());
        couponType.setUpdatedAt(record.getUpdatedAt());
    }
}
