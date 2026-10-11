package com.metawebthree.promotion.infrastructure.persistence.repository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.exception.PromotionErrorCode;
import com.metawebthree.promotion.domain.exception.PromotionException;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponConstants;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CouponMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CouponRecord;

@Repository
public class MybatisCouponRepository implements CouponRepository {
    private final CouponMapper mapper;

    public MybatisCouponRepository(CouponMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public void save(Coupon coupon) {
        try {
            mapper.insert(toRecord(coupon));
        } catch (DuplicateKeyException ex) {
            throw new PromotionException(PromotionErrorCode.CONFLICT, "duplicate coupon code");
        }
    }

    @Override
    public void saveAll(List<Coupon> coupons) {
        for (Coupon coupon : coupons) {
            save(coupon);
        }
    }

    @Override
    public Coupon findByCode(String code) {
        CouponRecord record = mapper.selectOne(new LambdaQueryWrapper<CouponRecord>()
                .eq(CouponRecord::getCode, code));
        return record == null ? null : toDomain(record);
    }

    @Override
    public Coupon findFirstAvailableByType(Long couponTypeId) {
        CouponRecord record = mapper.selectOne(buildAvailableQuery(couponTypeId)
                .orderByAsc(CouponRecord::getCreatedAt)
                .last("limit 1"));
        return record == null ? null : toDomain(record);
    }

    @Override
    public List<Coupon> findAvailableByType(Long couponTypeId, int limit) {
        List<CouponRecord> records = mapper.selectList(buildAvailableQuery(couponTypeId)
                .orderByAsc(CouponRecord::getCreatedAt)
                .last("limit " + limit));
        return toDomainList(records);
    }

    @Override
    public void updateOwnerIfAvailable(Long couponId, Long ownerUserId, Integer acquireMethod) {
        LambdaUpdateWrapper<CouponRecord> update = new LambdaUpdateWrapper<CouponRecord>()
                .eq(CouponRecord::getId, couponId)
                .eq(CouponRecord::getOwnerUserId, CouponConstants.UNASSIGNED_OWNER_USER_ID)
                .eq(CouponRecord::getUseStatus, CouponStatus.UNUSED.getCode())
                .set(CouponRecord::getOwnerUserId, ownerUserId)
                .set(CouponRecord::getAcquireMethod, acquireMethod)
                .set(CouponRecord::getUpdatedAt, LocalDateTime.now());
        ensureUpdated(mapper.update(null, update), "coupon not available");
    }

    @Override
    public void updateOwnerForCode(String code, Long ownerUserId, Integer acquireMethod, Integer transferStatus) {
        LambdaUpdateWrapper<CouponRecord> update = new LambdaUpdateWrapper<CouponRecord>()
                .eq(CouponRecord::getCode, code)
                .eq(CouponRecord::getUseStatus, CouponStatus.UNUSED.getCode())
                .set(CouponRecord::getOwnerUserId, ownerUserId)
                .set(CouponRecord::getAcquireMethod, acquireMethod)
                .set(CouponRecord::getTransferStatus, transferStatus)
                .set(CouponRecord::getUpdatedAt, LocalDateTime.now());
        ensureUpdated(mapper.update(null, update), "coupon not available");
    }

    @Override
    public void updateStatusToUsed(String code, Long ownerUserId, String orderNo, String consumerName, String operatorName) {
        LambdaUpdateWrapper<CouponRecord> update = new LambdaUpdateWrapper<CouponRecord>()
                .eq(CouponRecord::getCode, code)
                .eq(CouponRecord::getUseStatus, CouponStatus.UNUSED.getCode());
        if (ownerUserId != null) {
            update.eq(CouponRecord::getOwnerUserId, ownerUserId);
        }
        update.set(CouponRecord::getUseStatus, CouponStatus.USED.getCode())
                .set(CouponRecord::getOrderNo, orderNo)
                .set(CouponRecord::getConsumerName, consumerName)
                .set(CouponRecord::getOperatorName, operatorName)
                .set(CouponRecord::getUsedAt, LocalDateTime.now())
                .set(CouponRecord::getUpdatedAt, LocalDateTime.now());
        ensureUpdated(mapper.update(null, update), "coupon not consumable");
    }

    @Override
    public void updateTransferStatus(String code, Long ownerUserId, Integer transferStatus) {
        LambdaUpdateWrapper<CouponRecord> update = new LambdaUpdateWrapper<CouponRecord>()
                .eq(CouponRecord::getCode, code)
                .eq(CouponRecord::getOwnerUserId, ownerUserId)
                .eq(CouponRecord::getUseStatus, CouponStatus.UNUSED.getCode())
                .set(CouponRecord::getTransferStatus, transferStatus)
                .set(CouponRecord::getUpdatedAt, LocalDateTime.now());
        ensureUpdated(mapper.update(null, update), "coupon not available");
    }

    @Override
    public List<Coupon> listByOwner(Long ownerUserId, Integer useStatus) {
        LambdaQueryWrapper<CouponRecord> query = new LambdaQueryWrapper<CouponRecord>()
                .eq(CouponRecord::getOwnerUserId, ownerUserId);
        if (useStatus != null) {
            query.eq(CouponRecord::getUseStatus, useStatus);
        }
        return toDomainList(mapper.selectList(query));
    }

    @Override
    public List<Coupon> listByBatch(String batchId) {
        List<CouponRecord> records = mapper.selectList(new LambdaQueryWrapper<CouponRecord>()
                .eq(CouponRecord::getBatchId, batchId));
        return toDomainList(records);
    }

    private LambdaQueryWrapper<CouponRecord> buildAvailableQuery(Long couponTypeId) {
        return new LambdaQueryWrapper<CouponRecord>()
                .eq(CouponRecord::getCouponTypeId, couponTypeId)
                .eq(CouponRecord::getOwnerUserId, CouponConstants.UNASSIGNED_OWNER_USER_ID)
                .eq(CouponRecord::getUseStatus, CouponStatus.UNUSED.getCode());
    }

    private CouponRecord toRecord(Coupon coupon) {
        CouponRecord record = new CouponRecord();
        copyRecordBasic(record, coupon);
        copyRecordTime(record, coupon);
        return record;
    }

    private Coupon toDomain(CouponRecord record) {
        Coupon coupon = new Coupon();
        copyDomainBasic(coupon, record);
        copyDomainTime(coupon, record);
        return coupon;
    }

    private List<Coupon> toDomainList(List<CouponRecord> records) {
        List<Coupon> coupons = new ArrayList<>();
        for (CouponRecord record : records) {
            coupons.add(toDomain(record));
        }
        return coupons;
    }

    private void copyRecordBasic(CouponRecord record, Coupon coupon) {
        record.setId(coupon.getId());
        record.setCode(coupon.getCode());
        record.setCouponTypeId(coupon.getCouponTypeId());
        record.setOwnerUserId(coupon.getOwnerUserId());
        record.setTransferStatus(coupon.getTransferStatus());
        record.setAcquireMethod(coupon.getAcquireMethod());
        record.setUseStatus(coupon.getUseStatus());
        record.setOrderNo(coupon.getOrderNo());
        record.setConsumerName(coupon.getConsumerName());
        record.setOperatorName(coupon.getOperatorName());
        record.setBatchId(coupon.getBatchId());
    }

    private void copyRecordTime(CouponRecord record, Coupon coupon) {
        record.setUsedAt(coupon.getUsedAt());
        record.setCreatedAt(coupon.getCreatedAt());
        record.setUpdatedAt(coupon.getUpdatedAt());
    }

    private void copyDomainBasic(Coupon coupon, CouponRecord record) {
        coupon.setId(record.getId());
        coupon.setCode(record.getCode());
        coupon.setCouponTypeId(record.getCouponTypeId());
        coupon.setOwnerUserId(record.getOwnerUserId());
        coupon.setTransferStatus(record.getTransferStatus());
        coupon.setAcquireMethod(record.getAcquireMethod());
        coupon.setUseStatus(record.getUseStatus());
        coupon.setOrderNo(record.getOrderNo());
        coupon.setConsumerName(record.getConsumerName());
        coupon.setOperatorName(record.getOperatorName());
        coupon.setBatchId(record.getBatchId());
    }

    private void copyDomainTime(Coupon coupon, CouponRecord record) {
        coupon.setUsedAt(record.getUsedAt());
        coupon.setCreatedAt(record.getCreatedAt());
        coupon.setUpdatedAt(record.getUpdatedAt());
    }

    private void ensureUpdated(int updated, String message) {
        if (updated != 1) {
            throw new PromotionException(PromotionErrorCode.CONFLICT, message);
        }
    }
}
