package com.metawebthree.promotion.application;

import java.time.LocalDateTime;

import com.metawebthree.promotion.domain.exception.PromotionErrorCode;
import com.metawebthree.promotion.domain.exception.PromotionException;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

public class CouponTypeCommandService {
    private final CouponTypeRepository couponTypeRepository;
    private final TimeProvider timeProvider;

    public CouponTypeCommandService(CouponTypeRepository couponTypeRepository, TimeProvider timeProvider) {
        this.couponTypeRepository = couponTypeRepository;
        this.timeProvider = timeProvider;
    }

    public void create(CouponType couponType) {
        validateTime(couponType);
        applyDefaultEnabled(couponType);
        applyCreateTime(couponType);
        couponTypeRepository.save(couponType);
    }

    private void validateTime(CouponType couponType) {
        if (couponType.getStartTime() == null || couponType.getEndTime() == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing time range");
        }
        if (!couponType.getStartTime().isBefore(couponType.getEndTime())) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid time range");
        }
    }

    private void applyDefaultEnabled(CouponType couponType) {
        if (couponType.getIsEnabled() == null) {
            couponType.setIsEnabled(Boolean.TRUE);
        }
    }

    private void applyCreateTime(CouponType couponType) {
        LocalDateTime now = timeProvider.now();
        if (couponType.getCreatedAt() == null) {
            couponType.setCreatedAt(now);
        }
        couponType.setUpdatedAt(now);
    }
}
