package com.metawebthree.common.dto;

import java.math.BigDecimal;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
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
