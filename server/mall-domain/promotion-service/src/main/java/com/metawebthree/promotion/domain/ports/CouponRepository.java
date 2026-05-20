package com.metawebthree.promotion.domain.ports;

import java.util.List;

import com.metawebthree.promotion.domain.model.Coupon;

public interface CouponRepository {
    void save(Coupon coupon);
    void saveAll(List<Coupon> coupons);
    Coupon findByCode(String code);
    Coupon findFirstAvailableByType(Long couponTypeId);
    List<Coupon> findAvailableByType(Long couponTypeId, int limit);
    void updateOwnerIfAvailable(Long couponId, Long ownerUserId, Integer acquireMethod);
    void updateOwnerForCode(String code, Long ownerUserId, Integer acquireMethod, Integer transferStatus);
    void updateStatusToUsed(String code, Long ownerUserId, String orderNo, String consumerName, String operatorName);
    void updateTransferStatus(String code, Long ownerUserId, Integer transferStatus);
    List<Coupon> listByOwner(Long ownerUserId, Integer useStatus);
    List<Coupon> listByBatch(String batchId);
}
