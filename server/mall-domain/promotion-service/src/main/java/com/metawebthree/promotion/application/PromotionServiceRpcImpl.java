package com.metawebthree.promotion.application;

import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.CouponProto;
import com.metawebthree.common.generated.rpc.CouponTypeProto;
import com.metawebthree.common.generated.rpc.GetUserCouponsRequest;
import com.metawebthree.common.generated.rpc.GetUserCouponsResponse;
import com.metawebthree.common.generated.rpc.ListCouponTypesByProductRequest;
import com.metawebthree.common.generated.rpc.ListCouponTypesByProductResponse;
import com.metawebthree.common.generated.rpc.ReturnCouponRequest;
import com.metawebthree.common.generated.rpc.ReturnCouponResponse;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.ports.CouponRepository;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class PromotionServiceRpcImpl implements com.metawebthree.common.generated.rpc.PromotionService {

    private final CouponQueryService couponQueryService;
    private final CouponCommandService couponCommandService;
    private final CouponRepository couponRepository;

    public PromotionServiceRpcImpl(CouponQueryService couponQueryService,
                                   CouponCommandService couponCommandService,
                                   CouponRepository couponRepository) {
        this.couponQueryService = couponQueryService;
        this.couponCommandService = couponCommandService;
        this.couponRepository = couponRepository;
    }

    @Override
    public ListCouponTypesByProductResponse listCouponTypesByProduct(ListCouponTypesByProductRequest request) {
        List<CouponType> types = couponQueryService.listByProduct(request.getProductId());

        ListCouponTypesByProductResponse.Builder responseBuilder = ListCouponTypesByProductResponse.newBuilder();

        for (CouponType type : types) {
            CouponTypeProto.Builder protoBuilder = CouponTypeProto.newBuilder()
                    .setId(type.getId())
                    .setName(type.getName() != null ? type.getName() : "")
                    .setDescription(type.getDescription() != null ? type.getDescription() : "")
                    .setImageUrl(type.getImageUrl() != null ? type.getImageUrl() : "")
                    .setMinimumOrderAmount(type.getMinimumOrderAmount() != null ? type.getMinimumOrderAmount().doubleValue() : 0.0)
                    .setDiscountAmount(type.getDiscountAmount() != null ? type.getDiscountAmount().doubleValue() : 0.0)
                    .setIsEnabled(Boolean.TRUE.equals(type.getIsEnabled()));

            if (type.getStartTime() != null) {
                protoBuilder.setStartTime(type.getStartTime().toEpochSecond(java.time.ZoneOffset.UTC));
            }
            if (type.getEndTime() != null) {
                protoBuilder.setEndTime(type.getEndTime().toEpochSecond(java.time.ZoneOffset.UTC));
            }

            responseBuilder.addCouponTypes(protoBuilder.build());
        }

        return responseBuilder.build();
    }

    @Override
    public CompletableFuture<ListCouponTypesByProductResponse> listCouponTypesByProductAsync(
            ListCouponTypesByProductRequest request) {
        return CompletableFuture.completedFuture(listCouponTypesByProduct(request));
    }

    @Override
    public ReturnCouponResponse returnCoupon(ReturnCouponRequest request) {
        ReturnCouponResponse.Builder responseBuilder = ReturnCouponResponse.newBuilder();
        try {
            List<Coupon> coupons = couponQueryService.listByOwner(request.getUserId(), CouponStatus.USED.getCode());
            Coupon targetCoupon = coupons.stream()
                    .filter(c -> c.getOrderNo() != null && c.getOrderNo().equals(String.valueOf(request.getOrderId())))
                    .findFirst()
                    .orElse(null);

            if (targetCoupon != null) {
                targetCoupon.setUseStatus(CouponStatus.UNUSED.getCode());
                targetCoupon.setOrderNo(null);
                targetCoupon.setUsedAt(null);
                targetCoupon.setUpdatedAt(LocalDateTime.now());
                couponRepository.save(targetCoupon);
                responseBuilder.setSuccess(true).setMessage("优惠券已返还");
                log.info("优惠券返还成功 - userId: {}, couponId: {}, orderId: {}", 
                        request.getUserId(), targetCoupon.getId(), request.getOrderId());
            } else {
                responseBuilder.setSuccess(false).setMessage("未找到该订单使用的优惠券");
            }
        } catch (Exception e) {
            log.error("优惠券返还失败 - userId: {}, orderId: {}, error: {}", 
                    request.getUserId(), request.getOrderId(), e.getMessage(), e);
            responseBuilder.setSuccess(false).setMessage("优惠券返还失败: " + e.getMessage());
        }
        return responseBuilder.build();
    }

    @Override
    public CompletableFuture<ReturnCouponResponse> returnCouponAsync(ReturnCouponRequest request) {
        return CompletableFuture.completedFuture(returnCoupon(request));
    }

    @Override
    public GetUserCouponsResponse getUserCoupons(GetUserCouponsRequest request) {
        GetUserCouponsResponse.Builder responseBuilder = GetUserCouponsResponse.newBuilder();
        try {
            Integer useStatus = request.getOnlyUnused() ? CouponStatus.UNUSED.getCode() : null;
            List<Coupon> coupons = couponQueryService.listByOwner(request.getUserId(), useStatus);
            
            for (Coupon coupon : coupons) {
                CouponProto.Builder protoBuilder = CouponProto.newBuilder()
                        .setId(coupon.getId())
                        .setCouponTypeId(coupon.getCouponTypeId() != null ? coupon.getCouponTypeId() : 0)
                        .setStatus(coupon.getUseStatus() != null ? CouponStatus.fromCode(coupon.getUseStatus()).name() : "UNKNOWN");
                
                if (coupon.getOrderNo() != null) {
                    try {
                        protoBuilder.setOrderId(Long.parseLong(coupon.getOrderNo()));
                    } catch (NumberFormatException ignored) {}
                }
                if (coupon.getUsedAt() != null) {
                    protoBuilder.setUseTime(coupon.getUsedAt().toEpochSecond(java.time.ZoneOffset.UTC));
                }
                if (coupon.getCreatedAt() != null) {
                    protoBuilder.setReceiveTime(coupon.getCreatedAt().toEpochSecond(java.time.ZoneOffset.UTC));
                }
                responseBuilder.addCoupons(protoBuilder.build());
            }
        } catch (Exception e) {
            log.error("获取用户优惠券失败 - userId: {}, error: {}", request.getUserId(), e.getMessage(), e);
        }
        return responseBuilder.build();
    }

    @Override
    public CompletableFuture<GetUserCouponsResponse> getUserCouponsAsync(GetUserCouponsRequest request) {
        return CompletableFuture.completedFuture(getUserCoupons(request));
    }
}
