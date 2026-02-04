package com.metawebthree.promotion.application;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import com.metawebthree.promotion.domain.enums.CouponMethod;
import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.enums.PassStatus;
import com.metawebthree.promotion.domain.exception.PromotionErrorCode;
import com.metawebthree.promotion.domain.exception.PromotionException;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponBatch;
import com.metawebthree.promotion.domain.model.CouponConstants;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CodeGenerator;
import com.metawebthree.promotion.domain.ports.CouponBatchRepository;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

public class CouponCommandService {
    private final CouponRepository couponRepository;
    private final CouponTypeRepository couponTypeRepository;
    private final CouponBatchRepository couponBatchRepository;
    private final CodeGenerator codeGenerator;
    private final TimeProvider timeProvider;
    private final CouponPolicy policy;

    public CouponCommandService(CouponRepository couponRepository, CouponTypeRepository couponTypeRepository,
            CouponBatchRepository couponBatchRepository, CodeGenerator codeGenerator,
            TimeProvider timeProvider, CouponPolicy policy) {
        this.couponRepository = couponRepository;
        this.couponTypeRepository = couponTypeRepository;
        this.couponBatchRepository = couponBatchRepository;
        this.codeGenerator = codeGenerator;
        this.timeProvider = timeProvider;
        this.policy = policy;
    }

    public void generateBatch(String batchId, Long couponTypeId, int count) {
        validateBatchRequest(batchId, count);
        CouponType type = loadType(couponTypeId);
        ensureTypeActive(type);
        CouponBatch batch = buildBatch(batchId, couponTypeId, count);
        couponBatchRepository.save(batch);
        List<Coupon> coupons = buildCoupons(batchId, couponTypeId, count);
        saveCouponsWithRetry(coupons);
    }

    public void assignCoupon(String code, Long ownerUserId) {
        validateAssignRequest(code, ownerUserId);
        couponRepository.updateOwnerForCode(code, ownerUserId,
                CouponMethod.ADMIN_ASSIGN.getCode(), PassStatus.CLOSED.getCode());
    }

    public void batchAssign(Long couponTypeId, List<Long> userIds, int amount) {
        validateBatchAssignRequest(userIds, amount);
        CouponType type = loadType(couponTypeId);
        ensureTypeActive(type);
        int total = userIds.size() * amount;
        List<Coupon> coupons = couponRepository.findAvailableByType(couponTypeId, total);
        ensureEnoughCoupons(coupons.size(), total);
        assignSequential(userIds, amount, coupons);
    }

    public void claim(Long couponTypeId, Long ownerUserId) {
        validateClaimRequest(couponTypeId, ownerUserId);
        CouponType type = loadType(couponTypeId);
        ensureTypeActive(type);
        Coupon coupon = couponRepository.findFirstAvailableByType(couponTypeId);
        ensureCouponExists(coupon);
        couponRepository.updateOwnerIfAvailable(coupon.getId(), ownerUserId,
                CouponMethod.SELF_CLAIM.getCode());
    }

    public void consume(String code, Long ownerUserId, String orderNo, String consumerName, String operatorName) {
        validateConsumeRequest(code, orderNo);
        couponRepository.updateStatusToUsed(code, ownerUserId, orderNo, consumerName, operatorName);
    }

    public void openTransfer(String code, Long ownerUserId) {
        validateTransferRequest(code, ownerUserId);
        couponRepository.updateTransferStatus(code, ownerUserId, PassStatus.OPEN.getCode());
    }

    public void closeTransfer(String code, Long ownerUserId) {
        validateTransferRequest(code, ownerUserId);
        couponRepository.updateTransferStatus(code, ownerUserId, PassStatus.CLOSED.getCode());
    }

    public void claimTransfer(String code, Long ownerUserId) {
        validateTransferRequest(code, ownerUserId);
        Coupon coupon = couponRepository.findByCode(code);
        ensureCouponExists(coupon);
        CouponType type = loadType(coupon.getCouponTypeId());
        ensureTypeActive(type);
        couponRepository.updateOwnerForCode(code, ownerUserId,
                CouponMethod.TRANSFER.getCode(), PassStatus.CLOSED.getCode());
    }

    private void validateBatchRequest(String batchId, int count) {
        if (batchId == null || batchId.isBlank()) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing batchId");
        }
        if (count < 1 || count > policy.getMaxGenerateCount()) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid count");
        }
    }

    private CouponType loadType(Long couponTypeId) {
        CouponType type = couponTypeRepository.findById(couponTypeId);
        if (type == null) {
            throw new PromotionException(PromotionErrorCode.NOT_FOUND, "coupon type not found");
        }
        return type;
    }

    private void ensureTypeActive(CouponType type) {
        LocalDateTime now = timeProvider.now();
        if (!Boolean.TRUE.equals(type.getIsEnabled())) {
            throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "coupon type disabled");
        }
        if (type.getStartTime().isAfter(now) || type.getEndTime().isBefore(now)) {
            throw new PromotionException(PromotionErrorCode.EXPIRED, "coupon type expired");
        }
    }

    private CouponBatch buildBatch(String batchId, Long couponTypeId, int count) {
        CouponBatch batch = new CouponBatch();
        batch.setId(batchId);
        batch.setCouponTypeId(couponTypeId);
        batch.setTotalCount(count);
        batch.setCreatedAt(timeProvider.now());
        return batch;
    }

    private List<Coupon> buildCoupons(String batchId, Long couponTypeId, int count) {
        List<Coupon> coupons = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            coupons.add(buildCoupon(batchId, couponTypeId));
        }
        return coupons;
    }

    private Coupon buildCoupon(String batchId, Long couponTypeId) {
        Coupon coupon = new Coupon();
        coupon.setCouponTypeId(couponTypeId);
        coupon.setOwnerUserId(CouponConstants.UNASSIGNED_OWNER_USER_ID);
        coupon.setTransferStatus(PassStatus.CLOSED.getCode());
        coupon.setAcquireMethod(CouponMethod.ADMIN_ASSIGN.getCode());
        coupon.setUseStatus(CouponStatus.UNUSED.getCode());
        coupon.setOrderNo(null);
        coupon.setConsumerName(null);
        coupon.setOperatorName(null);
        coupon.setBatchId(batchId);
        coupon.setCreatedAt(timeProvider.now());
        coupon.setUpdatedAt(timeProvider.now());
        return coupon;
    }

    private void saveCouponsWithRetry(List<Coupon> coupons) {
        for (Coupon coupon : coupons) {
            saveCouponWithRetry(coupon);
        }
    }

    private void saveCouponWithRetry(Coupon coupon) {
        int attempts = 0;
        while (attempts < policy.getRetryLimit()) {
            coupon.setCode(codeGenerator.nextCode());
            try {
                couponRepository.save(coupon);
                return;
            } catch (PromotionException ex) {
                if (ex.getErrorCode() != PromotionErrorCode.CONFLICT) {
                    throw ex;
                }
            }
            attempts++;
        }
        throw new PromotionException(PromotionErrorCode.CONFLICT, "failed to create coupon code");
    }

    private void validateAssignRequest(String code, Long ownerUserId) {
        if (code == null || code.isBlank() || ownerUserId == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid assign request");
        }
    }

    private void validateBatchAssignRequest(List<Long> userIds, int amount) {
        if (userIds == null || userIds.isEmpty() || amount < 1) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid assign request");
        }
    }

    private void ensureEnoughCoupons(int available, int required) {
        if (available < required) {
            throw new PromotionException(PromotionErrorCode.CONFLICT, "not enough coupons");
        }
    }

    private void assignSequential(List<Long> userIds, int amount, List<Coupon> coupons) {
        int index = 0;
        for (Long userId : userIds) {
            for (int j = 0; j < amount; j++) {
                Coupon coupon = coupons.get(index++);
                couponRepository.updateOwnerIfAvailable(coupon.getId(), userId,
                        CouponMethod.ADMIN_ASSIGN.getCode());
            }
        }
    }

    private void validateClaimRequest(Long couponTypeId, Long ownerUserId) {
        if (couponTypeId == null || ownerUserId == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid claim request");
        }
    }

    private void ensureCouponExists(Coupon coupon) {
        if (coupon == null) {
            throw new PromotionException(PromotionErrorCode.NOT_FOUND, "coupon not found");
        }
    }

    private void validateConsumeRequest(String code, String orderNo) {
        if (code == null || code.isBlank() || orderNo == null || orderNo.isBlank()) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid consume request");
        }
    }

    private void validateTransferRequest(String code, Long ownerUserId) {
        if (code == null || code.isBlank() || ownerUserId == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid transfer request");
        }
    }

}
