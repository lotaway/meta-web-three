package com.metawebthree.aftersale.application.dto;

import lombok.Data;

@Data
public class AfterSaleQueryDTO {
    private Integer pageNum = 1;
    private Integer pageSize = 10;
    private Integer status;
    private Integer type;
    private String orderNo;
    private String userId;
    private String startDate;
    private String endDate;
}