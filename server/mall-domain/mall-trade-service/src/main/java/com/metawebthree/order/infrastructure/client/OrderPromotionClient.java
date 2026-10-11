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
public class OrderPromotionClient {

    @DubboReference(check = false, lazy = true)
    private PromotionService promotionService;

    /**
     * 杩旇繕鐢ㄦ埛浣跨敤鐨勪紭鎯犲埜
     * @param userId 鐢ㄦ埛ID
     * @param couponId 浼樻儬鍒窱D
     * @param orderId 璁㈠崟ID
     * @return 鏄惁鎴愬姛
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
                log.info("浼樻儬鍒歌繑杩樻垚鍔?- userId: {}, couponId: {}, orderId: {}", userId, couponId, orderId);
                return true;
            } else {
                log.warn("浼樻儬鍒歌繑杩樺け璐?- userId: {}, couponId: {}, orderId: {}, message: {}", 
                        userId, couponId, orderId, response.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("浼樻儬鍒歌繑杩樺紓甯?- userId: {}, couponId: {}, orderId: {}, error: {}", 
                    userId, couponId, orderId, e.getMessage(), e);
            return false;
        }
    }

    /**
     * 鑾峰彇鐢ㄦ埛鐨勪紭鎯犲埜鍒楄〃
     * @param userId 鐢ㄦ埛ID
     * @param onlyUnused 鏄惁鍙繑鍥炴湭浣跨敤鐨勪紭鎯犲埜
     * @return 浼樻儬鍒告暟閲?
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
            log.error("鑾峰彇鐢ㄦ埛浼樻儬鍒稿垪琛ㄥけ璐?- userId: {}, error: {}", userId, e.getMessage(), e);
            return 0;
        }
    }
}
