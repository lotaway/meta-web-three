package com.metawebthree.promotion.application;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.ArrayList;

import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.exception.PromotionErrorCode;
import com.metawebthree.promotion.domain.exception.PromotionException;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;
import com.metawebthree.promotion.domain.ports.MerkleService;

public class CouponQueryService {
    private final CouponRepository couponRepository;
    private final CouponTypeRepository couponTypeRepository;
    private final TimeProvider timeProvider;
    private final MerkleService merkleService;

    public CouponQueryService(CouponRepository couponRepository, CouponTypeRepository couponTypeRepository,
            TimeProvider timeProvider, MerkleService merkleService) {
        this.couponRepository = couponRepository;
        this.couponTypeRepository = couponTypeRepository;
        this.timeProvider = timeProvider;
        this.merkleService = merkleService;
    }

    public List<String> getMerkleProof(Coupon coupon) {
        if (coupon == null || coupon.getBatchId() == null) return null;
        List<Coupon> batchCoupons = couponRepository.listByBatch(coupon.getBatchId());
        CouponType type = couponTypeRepository.findById(coupon.getCouponTypeId());
        List<byte[]> leaves = new ArrayList<>();
        byte[] targetLeaf = null;
        for (Coupon c : batchCoupons) {
            byte[] leaf = computeLeaf(c, type);
            if (leaf == null) continue;
            leaves.add(leaf);
            if (c.getCode().equals(coupon.getCode())) targetLeaf = leaf;
        }
        return (targetLeaf != null) ? merkleService.getMerkleProof(leaves, targetLeaf) : null;
    }

    private byte[] computeLeaf(Coupon c, CouponType type) {
        String wallet = c.getOwnerWalletAddress();
        if (wallet == null || wallet.isBlank()) return null;
        return merkleService.computeLeaf(
                wallet,
                c.getCode(),
                type.getDiscountAmount().longValue(),
                type.getMinimumOrderAmount().longValue(),
                type.getStartTime().toEpochSecond(java.time.ZoneOffset.UTC),
                type.getEndTime().toEpochSecond(java.time.ZoneOffset.UTC));
    }

    public List<Coupon> listByOwner(Long ownerUserId, Integer useStatus) {
        if (ownerUserId == null) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing ownerUserId");
        return couponRepository.listByOwner(ownerUserId, useStatus);
    }

    public List<Coupon> listByBatch(String batchId) {
        if (batchId == null || batchId.isBlank()) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing batchId");
        return couponRepository.listByBatch(batchId);
    }

    public CouponType getCouponType(Long id) {
        return couponTypeRepository.findById(id);
    }

    public List<CouponType> listByProduct(Long productId) {
        if (productId == null || productId <= 0) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing productId");
        }
        // 当前实现：返回所有在有效期内且已启用的通用券种
        // 未来可扩展为：叠加 product/category 维度的精准适配
        return couponTypeRepository.listEnabledActive(timeProvider.now());
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
        if (code == null || code.isBlank() || ownerUserId == null || orderAmount == null) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid request");
        if (orderAmount.compareTo(BigDecimal.ZERO) < 0) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid amount");
    }

    private Coupon loadCoupon(String code) {
        Coupon coupon = couponRepository.findByCode(code);
        if (coupon == null) throw new PromotionException(PromotionErrorCode.NOT_FOUND, "not found");
        return coupon;
    }

    private void ensureOwned(Coupon coupon, Long ownerUserId) {
        if (!ownerUserId.equals(coupon.getOwnerUserId())) throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "not owned");
    }

    private void ensureUnused(Coupon coupon) {
        if (coupon.getUseStatus() != CouponStatus.UNUSED.getCode()) throw new PromotionException(PromotionErrorCode.CONFLICT, "already used");
    }

    private CouponType loadType(Long typeId) {
        CouponType type = couponTypeRepository.findById(typeId);
        if (type == null) throw new PromotionException(PromotionErrorCode.NOT_FOUND, "type not found");
        return type;
    }

    private void ensureActive(CouponType type) {
        if (!Boolean.TRUE.equals(type.getIsEnabled())) throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "disabled");
        LocalDateTime now = timeProvider.now();
        if (type.getStartTime().isAfter(now) || type.getEndTime().isBefore(now)) throw new PromotionException(PromotionErrorCode.EXPIRED, "expired");
    }

    private void ensureAmount(CouponType type, BigDecimal orderAmount) {
        if (type.getMinimumOrderAmount().compareTo(orderAmount) > 0) throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "amount ineligible");
    }

    private BigDecimal computePayable(CouponType type, BigDecimal orderAmount, BigDecimal deliveryFee) {
        BigDecimal fee = (deliveryFee == null) ? BigDecimal.ZERO : deliveryFee;
        if (fee.compareTo(BigDecimal.ZERO) < 0) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid fee");
        BigDecimal result = orderAmount.add(fee).subtract(type.getDiscountAmount());
        return result.compareTo(BigDecimal.ZERO) < 0 ? BigDecimal.ZERO : result;
    }

    public record CouponValidateResult(String couponTypeName, BigDecimal discountAmount, BigDecimal payableAmount) {}
}
