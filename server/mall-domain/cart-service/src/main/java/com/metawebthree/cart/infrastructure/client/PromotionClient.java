package com.metawebthree.cart.infrastructure.client;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.CouponTypeProto;
import com.metawebthree.common.generated.rpc.ListCouponTypesByProductRequest;
import com.metawebthree.common.generated.rpc.ListCouponTypesByProductResponse;
import com.metawebthree.common.generated.rpc.PromotionService;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class PromotionClient {

    @DubboReference(check = false, lazy = true)
    private PromotionService promotionService;

    public List<PromotionInfo> getPromotionsByProductId(Long productId) {
        List<PromotionInfo> promotions = new ArrayList<>();

        try {
            ListCouponTypesByProductRequest request = ListCouponTypesByProductRequest.newBuilder()
                    .setProductId(productId)
                    .build();

            ListCouponTypesByProductResponse response = promotionService.listCouponTypesByProduct(request);

            if (response == null) {
                return promotions;
            }

            for (CouponTypeProto couponType : response.getCouponTypesList()) {
                PromotionInfo info = new PromotionInfo();
                info.setPromotionType("coupon");
                info.setPromotionTag(couponType.getName());
                info.setDiscountAmount(BigDecimal.valueOf(couponType.getDiscountAmount()));
                info.setMinimumOrderAmount(BigDecimal.valueOf(couponType.getMinimumOrderAmount()));
                promotions.add(info);
            }
        } catch (Exception e) {
            log.warn("查询商品促销信息失败 - 商品ID: {}, 错误: {}", productId, e.getMessage());
        }

        return promotions;
    }

    public static class PromotionInfo {
        private String promotionType;
        private String promotionTag;
        private BigDecimal discountAmount;
        private BigDecimal minimumOrderAmount;

        public String getPromotionType() {
            return promotionType;
        }

        public void setPromotionType(String promotionType) {
            this.promotionType = promotionType;
        }

        public String getPromotionTag() {
            return promotionTag;
        }

        public void setPromotionTag(String promotionTag) {
            this.promotionTag = promotionTag;
        }

        public BigDecimal getDiscountAmount() {
            return discountAmount;
        }

        public void setDiscountAmount(BigDecimal discountAmount) {
            this.discountAmount = discountAmount;
        }

        public BigDecimal getMinimumOrderAmount() {
            return minimumOrderAmount;
        }

        public void setMinimumOrderAmount(BigDecimal minimumOrderAmount) {
            this.minimumOrderAmount = minimumOrderAmount;
        }
    }
}
