package com.metawebthree.dataanalysis.application.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RealTimeDashboardDTO {

    private BigDecimal todaySales;
    private BigDecimal todayOrders;
    private BigDecimal todayVisitors;
    private BigDecimal conversionRate;

    private BigDecimal todayProfit;
    private Integer pendingOrders;
    private Integer lowStockAlerts;
    private Integer pendingPayments;

    private List<HotProductDTO> hotProducts;
    private List<SalesByHourDTO> salesByHour;
    private Map<String, Integer> orderStatusDistribution;
    private Map<String, BigDecimal> categorySalesDistribution;

    private BigDecimal weekOverWeekGrowth;
    private BigDecimal monthOverMonthGrowth;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class HotProductDTO {
        private String productId;
        private String productName;
        private Integer salesCount;
        private BigDecimal salesAmount;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SalesByHourDTO {
        private Integer hour;
        private BigDecimal sales;
        private Integer orders;
    }
}