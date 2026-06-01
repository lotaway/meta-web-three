package com.metawebthree.aftersale.application.dto;

import lombok.Data;

@Data
public class AfterSaleStatisticDTO {
    private Long totalCount;
    private Long pendingCount;
    private Long processingCount;
    private Long approvedCount;
    private Long rejectedCount;
    private Long completedCount;
    private Long totalRefundAmount;
    private Long todayCount;
    private Long weekCount;
    private Long monthCount;
}