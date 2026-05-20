package com.metawebthree.promotion.application;

import java.time.LocalDateTime;

import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

public class CouponTypeQueryService {
    private final CouponTypeRepository couponTypeRepository;
    private final TimeProvider timeProvider;

    public CouponTypeQueryService(CouponTypeRepository couponTypeRepository, TimeProvider timeProvider) {
        this.couponTypeRepository = couponTypeRepository;
        this.timeProvider = timeProvider;
    }

    public CouponType getById(Long id) {
        return couponTypeRepository.findById(id);
    }

    public boolean isActive(CouponType couponType) {
        if (couponType == null || !Boolean.TRUE.equals(couponType.getIsEnabled())) {
            return false;
        }
        LocalDateTime now = timeProvider.now();
        return !couponType.getStartTime().isAfter(now) && !couponType.getEndTime().isBefore(now);
    }
}
