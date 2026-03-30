package com.metawebthree.order.domain.model;

import lombok.Builder;
import lombok.Getter;
import java.time.LocalDateTime;

@Getter
@Builder
public class OrderReturnApply {
    private final Long id;
    private final Long orderId;
    private final Long companyAddressId;
    private final Long productId;
    private final String orderNo;
    private final LocalDateTime createTime;
    private final String memberUsername;
    private final Double returnAmount;
    private final String returnName;
    private final String returnPhone;
    private final Integer status;
    private final LocalDateTime handleTime;
    private final String productPic;
    private final String productName;
    private final String productBrand;
    private final String productAttr;
    private final Integer productCount;
    private final Double productPrice;
    private final Double productRealPrice;
    private final String reason;
    private final String description;
    private final String proofPics;
    private final String handleNote;
    private final String handleMan;
    private final String receiveMan;
    private final LocalDateTime receiveTime;
    private final String receiveNote;

    public boolean isPending() {
        return status != null && status == 0;
    }

    public boolean isCompleted() {
        return status != null && status == 2;
    }
}
