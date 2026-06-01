package com.metawebthree.aftersale.application.dto;

import lombok.Data;

@Data
public class AfterSaleProcessDTO {
    private Long id;
    private Integer status;
    private String rejectReason;
    private String remark;
}