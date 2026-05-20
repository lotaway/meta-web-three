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
import com.metawebthree.promotion.domain.ports.BlockchainService;
import com.metawebthree.promotion.domain.ports.UserWalletService;
import com.metawebthree.promotion.domain.ports.MerkleService;

public class CouponCommandService {
    private final CouponRepository couponRepository;
    private final CouponTypeRepository couponTypeRepository;
    private final CouponBatchRepository couponBatchRepository;
    private final CodeGenerator codeGenerator;
    private final TimeProvider timeProvider;
    private final BlockchainService blockchainService;
    private final UserWalletService userWalletService;
    private final MerkleService merkleService;
    private final CouponPolicy policy;

    public CouponCommandService(CouponRepository couponRepository, CouponTypeRepository couponTypeRepository,
            CouponBatchRepository couponBatchRepository, CodeGenerator codeGenerator,
            TimeProvider timeProvider, BlockchainService blockchainService,
            UserWalletService userWalletService, MerkleService merkleService, CouponPolicy policy) {
        this.couponRepository = couponRepository;
        this.couponTypeRepository = couponTypeRepository;
        this.couponBatchRepository = couponBatchRepository;
        this.codeGenerator = codeGenerator;
        this.timeProvider = timeProvider;
        this.blockchainService = blockchainService;
        this.userWalletService = userWalletService;
        this.merkleService = merkleService;
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

    public void publishBatchRoot(String batchId) {
        CouponBatch batch = loadBatch(batchId);
        List<Coupon> coupons = couponRepository.listByBatch(batchId);
        CouponType type = loadType(batch.getCouponTypeId());
        List<byte[]> leaves = buildLeavesFromCoupons(coupons, type);
        publishRootToChain(batch, leaves);
    }

    private List<byte[]> buildLeavesFromCoupons(List<Coupon> coupons, CouponType type) {
        List<byte[]> leaves = new ArrayList<>();
        for (Coupon coupon : coupons) {
            String wallet = coupon.getOwnerWalletAddress();
            if (wallet == null || wallet.isBlank()) continue;
            leaves.add(computeCouponLeaf(coupon, type, wallet));
        }
        if (leaves.isEmpty()) throw new PromotionException(PromotionErrorCode.CONFLICT, "no assigned coupons");
        return leaves;
    }

    private byte[] computeCouponLeaf(Coupon coupon, CouponType type, String wallet) {
        return merkleService.computeLeaf(
                wallet,
                coupon.getCode(),
                type.getDiscountAmount().longValue(),
                type.getMinimumOrderAmount().longValue(),
                type.getStartTime().toEpochSecond(java.time.ZoneOffset.UTC),
                type.getEndTime().toEpochSecond(java.time.ZoneOffset.UTC));
    }

    private void publishRootToChain(CouponBatch batch, List<byte[]> leaves) {
        String root = merkleService.getMerkleRoot(leaves);
        blockchainService.setCouponBatchRoot(batch.getId(), root);
        batch.setMerkleRoot(root);
        couponBatchRepository.save(batch);
    }

    public void assignCoupon(String code, Long ownerUserId) {
        validateAssignRequest(code, ownerUserId);
        String wallet = userWalletService.getWalletAddressByUserId(ownerUserId);
        couponRepository.updateOwnerForCode(code, ownerUserId,
                CouponMethod.ADMIN_ASSIGN.getCode(), PassStatus.CLOSED.getCode());
        updateCouponWallet(code, wallet);
    }

    private void updateCouponWallet(String code, String wallet) {
        Coupon coupon = couponRepository.findByCode(code);
        if (coupon != null) {
            coupon.setOwnerWalletAddress(wallet);
            couponRepository.save(coupon);
        }
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
        String wallet = userWalletService.getWalletAddressByUserId(ownerUserId);
        couponRepository.updateOwnerIfAvailable(coupon.getId(), ownerUserId,
                CouponMethod.SELF_CLAIM.getCode());
        updateCouponWallet(coupon.getCode(), wallet);
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
        ensureTypeActive(loadType(coupon.getCouponTypeId()));
        String wallet = userWalletService.getWalletAddressByUserId(ownerUserId);
        couponRepository.updateOwnerForCode(code, ownerUserId,
                CouponMethod.TRANSFER.getCode(), PassStatus.CLOSED.getCode());
        updateCouponWallet(code, wallet);
    }

    private void validateBatchRequest(String batchId, int count) {
        if (batchId == null || batchId.isBlank()) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing batchId");
        if (count < 1 || count > policy.getMaxGenerateCount()) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid count");
    }

    private CouponType loadType(Long couponTypeId) {
        CouponType type = couponTypeRepository.findById(couponTypeId);
        if (type == null) throw new PromotionException(PromotionErrorCode.NOT_FOUND, "type not found");
        return type;
    }

    private CouponBatch loadBatch(String batchId) {
        CouponBatch batch = couponBatchRepository.findById(batchId);
        if (batch == null) throw new PromotionException(PromotionErrorCode.NOT_FOUND, "batch not found");
        return batch;
    }

    private void ensureTypeActive(CouponType type) {
        if (!Boolean.TRUE.equals(type.getIsEnabled())) throw new PromotionException(PromotionErrorCode.NOT_ALLOWED, "type disabled");
        LocalDateTime now = timeProvider.now();
        if (type.getStartTime().isAfter(now) || type.getEndTime().isBefore(now)) throw new PromotionException(PromotionErrorCode.EXPIRED, "type expired");
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
        for (int i = 0; i < count; i++) coupons.add(buildCoupon(batchId, couponTypeId));
        return coupons;
    }

    private Coupon buildCoupon(String batchId, Long couponTypeId) {
        Coupon coupon = new Coupon();
        coupon.setCouponTypeId(couponTypeId);
        coupon.setOwnerUserId(CouponConstants.UNASSIGNED_OWNER_USER_ID);
        coupon.setTransferStatus(PassStatus.CLOSED.getCode());
        coupon.setAcquireMethod(CouponMethod.ADMIN_ASSIGN.getCode());
        coupon.setUseStatus(CouponStatus.UNUSED.getCode());
        coupon.setBatchId(batchId);
        coupon.setCreatedAt(timeProvider.now());
        coupon.setUpdatedAt(timeProvider.now());
        return coupon;
    }

    private void saveCouponsWithRetry(List<Coupon> coupons) {
        for (Coupon coupon : coupons) saveCouponWithRetry(coupon);
    }

    private void saveCouponWithRetry(Coupon coupon) {
        int attempts = 0;
        while (attempts < policy.getRetryLimit()) {
            coupon.setCode(codeGenerator.nextCode());
            try {
                couponRepository.save(coupon);
                return;
            } catch (PromotionException ex) {
                if (ex.getErrorCode() != PromotionErrorCode.CONFLICT) throw ex;
            }
            attempts++;
        }
        throw new PromotionException(PromotionErrorCode.CONFLICT, "retry limit reached");
    }

    private void validateAssignRequest(String code, Long ownerUserId) {
        if (code == null || code.isBlank() || ownerUserId == null) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid request");
    }

    private void validateBatchAssignRequest(List<Long> userIds, int amount) {
        if (userIds == null || userIds.isEmpty() || amount < 1) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid request");
    }

    private void ensureEnoughCoupons(int available, int required) {
        if (available < required) throw new PromotionException(PromotionErrorCode.CONFLICT, "not enough coupons");
    }

    private void assignSequential(List<Long> userIds, int amount, List<Coupon> coupons) {
        int index = 0;
        for (Long userId : userIds) {
            String wallet = userWalletService.getWalletAddressByUserId(userId);
            for (int j = 0; j < amount; j++) {
                Coupon coupon = coupons.get(index++);
                couponRepository.updateOwnerIfAvailable(coupon.getId(), userId, CouponMethod.ADMIN_ASSIGN.getCode());
                updateCouponWallet(coupon.getCode(), wallet);
            }
        }
    }

    private void validateClaimRequest(Long couponTypeId, Long ownerUserId) {
        if (couponTypeId == null || ownerUserId == null) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid request");
    }

    private void ensureCouponExists(Coupon coupon) {
        if (coupon == null) throw new PromotionException(PromotionErrorCode.NOT_FOUND, "coupon not found");
    }

    private void validateConsumeRequest(String code, String orderNo) {
        if (code == null || code.isBlank() || orderNo == null || orderNo.isBlank()) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid request");
    }

    private void validateTransferRequest(String code, Long ownerUserId) {
        if (code == null || code.isBlank() || ownerUserId == null) throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid request");
    }
}
