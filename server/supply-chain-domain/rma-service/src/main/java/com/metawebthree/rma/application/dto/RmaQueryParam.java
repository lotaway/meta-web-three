package com.metawebthree.rma.application.dto;

import lombok.Data;

@Data
public class RmaQueryParam {
    private String status;
    private String rmaNo;
    private String orderNo;
    private Integer pageNum = 1;
    private Integer pageSize = 20;
}
