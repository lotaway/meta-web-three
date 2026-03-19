package com.metawebthree.promotion.interfaces.web.dto;

import jakarta.validation.constraints.NotBlank;

public class CouponConsumeRequest {
    @NotBlank
    private String code;
    @NotBlank
    private String orderNo;
    private String consumerName;
    private String operatorName;

    public String getCode() { return code; }
    public void setCode(String code) { this.code = code; }
    public String getOrderNo() { return orderNo; }
    public void setOrderNo(String orderNo) { this.orderNo = orderNo; }
    public String getConsumerName() { return consumerName; }
    public void setConsumerName(String consumerName) { this.consumerName = consumerName; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
}
