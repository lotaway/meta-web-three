package com.metawebthree.promotion.application;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.exception.PromotionErrorCode;
import com.metawebthree.promotion.domain.exception.PromotionException;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

public class CouponQueryService {
    private final CouponRepository couponRepository;
    private final CouponTypeRepository couponTypeRepository;
    private final TimeProvider timeProvider;

    public CouponQueryService(CouponRepository couponRepository, CouponTypeRepository couponTypeRepository,
            TimeProvider timeProvider) {
        this.couponRepository = couponRepository;
        this.couponTypeRepository = couponTypeRepository;
        this.timeProvider = timeProvider;
    }

    public List<Coupon> listByOwner(Long ownerUserId, Integer useStatus) {
        if (ownerUserId == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing ownerUserId");
        }
        return couponRepository.listByOwner(ownerUserId, useStatus);
    }

    public List<Coupon> listByBatch(String batchId) {
        if (batchId == null || batchId.isBlank()) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing batchId");
        }
        return couponRepository.listByBatch(batchId);
    }

    public CouponType getCouponType(Long id) {
        return couponTypeRepository.findById(id);
    }

    public CouponValidateResult validate(String code, Long ownerUserId, BigDecimal orderAmount, BigDecimal deliveryFee) {
        validateRequest(code, ownerUserId, orderAmount);
        Coupon coupon = loadCoupon(code);
        ensureOwned(coupon, ownerUserId);
        ensureUnused(coupon);
        CouponType type = loadType(coupon.getCouponTypeId());
        ensureActive(type);
        ensureAmount(type, orderAmount);
        BigDecimal payable = computePayable(type, orderAmount, deliveryFee);
        return new CouponValidateResult(type.getName(), type.getDiscountAmount(), payable);
    }

    private void validateRequest(String code, Long ownerUserId, BigDecimal orderAmount) {
        if (code == null || code.isBlank() || ownerUserId == null || orderAmount == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid validate request");
        }
        if (orderAmount.compareTo(BigDecimal.ZERO) < 0) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid order amount");
        }
    }

    private Coupon loadCoupon(String code) {
        Coupon coupon = couponRepository.findByCode(code);
        if (coupon == null) {
            throw new PromotionException(PromotionErrorCode.NOT_FOUND, "coupon not found");
        }
        return coupon;
    }

    private void ensureOwned(Coupon coupon, Long ownerUserId) {
        if (!ownerUserId.equals(coupon.getOwnerUserId())) {
            throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "coupon not owned");
        }
    }

    private void ensureUnused(Coupon coupon) {
        if (coupon.getUseStatus() != CouponStatus.UNUSED.getCode()) {
            throw new PromotionException(PromotionErrorCode.CONFLICT, "coupon already used");
        }
    }

    private CouponType loadType(Long typeId) {
        CouponType type = couponTypeRepository.findById(typeId);
        if (type == null) {
            throw new PromotionException(PromotionErrorCode.NOT_FOUND, "coupon type not found");
        }
        return type;
    }

    private void ensureActive(CouponType type) {
        if (!Boolean.TRUE.equals(type.getIsEnabled())) {
            throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "coupon type disabled");
        }
        LocalDateTime now = timeProvider.now();
        if (type.getStartTime().isAfter(now) || type.getEndTime().isBefore(now)) {
            throw new PromotionException(PromotionErrorCode.EXPIRED, "coupon not in valid time");
        }
    }

    private void ensureAmount(CouponType type, BigDecimal orderAmount) {
        if (type.getMinimumOrderAmount().compareTo(orderAmount) > 0) {
            throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "order amount not eligible");
        }
    }

    private BigDecimal computePayable(CouponType type, BigDecimal orderAmount, BigDecimal deliveryFee) {
        BigDecimal fee = normalizeFee(deliveryFee);
        BigDecimal payable = orderAmount.add(fee).subtract(type.getDiscountAmount());
        return payable.compareTo(BigDecimal.ZERO) < 0 ? BigDecimal.ZERO : payable;
    }

    private BigDecimal normalizeFee(BigDecimal deliveryFee) {
        if (deliveryFee == null) {
            return BigDecimal.ZERO;
        }
        if (deliveryFee.compareTo(BigDecimal.ZERO) < 0) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid delivery fee");
        }
        return deliveryFee;
    }

    public static class CouponValidateResult {
        private final String couponTypeName;
        private final BigDecimal discountAmount;
        private final BigDecimal payableAmount;

        public CouponValidateResult(String couponTypeName, BigDecimal discountAmount, BigDecimal payableAmount) {
            this.couponTypeName = couponTypeName;
            this.discountAmount = discountAmount;
            this.payableAmount = payableAmount;
        }

        public String getCouponTypeName() { return couponTypeName; }
        public BigDecimal getDiscountAmount() { return discountAmount; }
        public BigDecimal getPayableAmount() { return payableAmount; }
    }
}
