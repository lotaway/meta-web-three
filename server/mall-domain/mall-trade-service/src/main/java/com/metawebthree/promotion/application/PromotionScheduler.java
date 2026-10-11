package com.metawebthree.promotion.application;

import java.time.LocalDateTime;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

/**
 * Promotion scheduler for automatic enable/disable based on time.
 * Runs every minute to check and update coupon type status.
 */
@Component
public class PromotionScheduler {
    private static final Logger log = LoggerFactory.getLogger(PromotionScheduler.class);
    
    private final CouponTypeRepository couponTypeRepository;
    private final TimeProvider timeProvider;

    public PromotionScheduler(CouponTypeRepository couponTypeRepository, TimeProvider timeProvider) {
        this.couponTypeRepository = couponTypeRepository;
        this.timeProvider = timeProvider;
    }

    /**
     * Scheduled task to update coupon type status based on time.
     * - Enable: when current time is within [startTime, endTime] and isEnabled is false
     * - Disable: when current time is before startTime and isEnabled is true
     * - Disable: when current time is after endTime and isEnabled is true
     */
    @Scheduled(cron = "0 * * * * ?") // Run every minute
    public void updatePromotionStatus() {
        LocalDateTime now = timeProvider.now();
        log.info("Starting promotion status update check at {}", now);
        
        List<CouponType> allCoupons = couponTypeRepository.listAll();
        int enabledCount = 0;
        int disabledCount = 0;
        
        for (CouponType coupon : allCoupons) {
            if (coupon.getStartTime() == null || coupon.getEndTime() == null) {
                continue;
            }
            
            boolean shouldBeEnabled = !now.isBefore(coupon.getStartTime()) && !now.isAfter(coupon.getEndTime());
            boolean currentEnabled = Boolean.TRUE.equals(coupon.getIsEnabled());
            
            if (shouldBeEnabled && !currentEnabled) {
                // Should be enabled but currently disabled
                coupon.setIsEnabled(true);
                coupon.setUpdatedAt(now);
                couponTypeRepository.update(coupon);
                enabledCount++;
                log.info("Enabled coupon type: {} (id={}), valid: {} - {}", 
                    coupon.getName(), coupon.getId(), coupon.getStartTime(), coupon.getEndTime());
            } else if (!shouldBeEnabled && currentEnabled) {
                // Should be disabled but currently enabled
                coupon.setIsEnabled(false);
                coupon.setUpdatedAt(now);
                couponTypeRepository.update(coupon);
                disabledCount++;
                log.info("Disabled coupon type: {} (id={}), valid: {} - {}", 
                    coupon.getName(), coupon.getId(), coupon.getStartTime(), coupon.getEndTime());
            }
        }
        
        log.info("Promotion status update completed. Enabled: {}, Disabled: {}", enabledCount, disabledCount);
    }
}