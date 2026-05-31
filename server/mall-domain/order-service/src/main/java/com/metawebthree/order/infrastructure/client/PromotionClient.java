package com.metawebthree.order.infrastructure.client;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.GetUserCouponsRequest;
import com.metawebthree.common.generated.rpc.GetUserCouponsResponse;
import com.metawebthree.common.generated.rpc.PromotionService;
import com.metawebthree.common.generated.rpc.ReturnCouponRequest;
import com.metawebthree.common.generated.rpc.ReturnCouponResponse;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class PromotionClient {

    @DubboReference(check = false, lazy = true)
    private PromotionService promotionService;

    /**
     * 返还用户使用的优惠券
     * @param userId 用户ID
     * @param couponId 优惠券ID
     * @param orderId 订单ID
     * @return 是否成功
     */
    public boolean returnCoupon(Long userId, Long couponId, Long orderId) {
        try {
            ReturnCouponRequest request = ReturnCouponRequest.newBuilder()
                    .setUserId(userId)
                    .setCouponId(couponId)
                    .setOrderId(orderId)
                    .build();

            ReturnCouponResponse response = promotionService.returnCoupon(request);
            if (response.getSuccess()) {
                log.info("优惠券返还成功 - userId: {}, couponId: {}, orderId: {}", userId, couponId, orderId);
                return true;
            } else {
                log.warn("优惠券返还失败 - userId: {}, couponId: {}, orderId: {}, message: {}", 
                        userId, couponId, orderId, response.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("优惠券返还异常 - userId: {}, couponId: {}, orderId: {}, error: {}", 
                    userId, couponId, orderId, e.getMessage(), e);
            return false;
        }
    }

    /**
     * 获取用户的优惠券列表
     * @param userId 用户ID
     * @param onlyUnused 是否只返回未使用的优惠券
     * @return 优惠券数量
     */
    public int getUserCouponCount(Long userId, boolean onlyUnused) {
        try {
            GetUserCouponsRequest request = GetUserCouponsRequest.newBuilder()
                    .setUserId(userId)
                    .setOnlyUnused(onlyUnused)
                    .build();

            GetUserCouponsResponse response = promotionService.getUserCoupons(request);
            return response.getCouponsCount();
        } catch (Exception e) {
            log.error("获取用户优惠券列表失败 - userId: {}, error: {}", userId, e.getMessage(), e);
            return 0;
        }
    }
}