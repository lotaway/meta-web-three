package com.metawebthree.order.domain.model;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("tb_order")
public class AdminOrderDO {
    private Long id;
    private Long memberId;
    private Long couponId;
    private String orderSn;
    private LocalDateTime createTime;
    private String memberUsername;
    private BigDecimal totalAmount;
    private BigDecimal payAmount;
    private BigDecimal freightAmount;
    private BigDecimal promotionAmount;
    private BigDecimal integrationAmount;
    private BigDecimal couponAmount;
    private BigDecimal discountAmount;
    private Integer payType;
    private Integer sourceType;
    private Integer status;
    private Integer orderType;
    private String deliveryCompany;
    private String deliverySn;
    private Integer autoConfirmDay;
    private Integer integration;
    private Integer growth;
    private String promotionInfo;
    private Integer billType;
    private String billHeader;
    private String billContent;
    private String billReceiverPhone;
    private String billReceiverEmail;
    private String receiverName;
    private String receiverPhone;
    private String receiverPostCode;
    private String receiverProvince;
    private String receiverCity;
    private String receiverRegion;
    private String receiverDetailAddress;
    private String note;
    private Integer confirmStatus;
    private Integer deleteStatus;
    private Long memberReceiveAddressId;
    private Integer useIntegration;
    private LocalDateTime paymentTime;
    private LocalDateTime deliveryTime;
    private LocalDateTime receiveTime;
    private LocalDateTime commentTime;
    private LocalDateTime modifyTime;
}
