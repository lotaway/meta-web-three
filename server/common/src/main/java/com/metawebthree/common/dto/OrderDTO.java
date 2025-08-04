package com.metawebthree.common.dto;

import java.math.BigDecimal;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class OrderDTO {
    private Long id;
    private Long userId;
    private String orderNo;
    private String orderStatus;
    private String orderType;
    private BigDecimal orderAmount;
    private String orderRemark;
}
