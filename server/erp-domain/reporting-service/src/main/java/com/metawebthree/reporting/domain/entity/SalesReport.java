package com.metawebthree.reporting.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class SalesReport {
    private Long id;
    private String reportNo;
    private ReportType type;
    private LocalDateTime reportDate;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
    private BigDecimal totalSalesAmount;
    private Integer totalOrderCount;
    private BigDecimal averageOrderAmount;
    private BigDecimal grossProfit;
    private BigDecimal profitMargin;
    private String categoryBreakdown;
    private String productRanking;
    private String channelBreakdown;
    private LocalDateTime createdAt;

    public enum ReportType {
        DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL
    }

    public void generateDailyReport(LocalDateTime reportDate) {
        this.reportNo = "SALES-" + reportDate.toLocalDate().toString().replace("-", "");
        this.type = ReportType.DAILY;
        this.reportDate = reportDate;
        this.startDate = reportDate.toLocalDate().atStartOfDay();
        this.endDate = reportDate.toLocalDate().atTime(23, 59, 59);
        this.createdAt = LocalDateTime.now();
    }

    public void generateMonthlyReport(int year, int month) {
        this.reportNo = "SALES-" + year + String.format("%02d", month);
        this.type = ReportType.MONTHLY;
        this.reportDate = LocalDateTime.of(year, month, 1, 0, 0).plusMonths(1).minusSeconds(1);
        this.startDate = LocalDateTime.of(year, month, 1, 0, 0);
        this.endDate = LocalDateTime.of(year, month, 1, 0, 0).plusMonths(1).minusSeconds(1);
        this.createdAt = LocalDateTime.now();
    }

    public void setMetrics(BigDecimal totalSalesAmount, Integer totalOrderCount,
                          BigDecimal grossProfit, BigDecimal profitMargin) {
        this.totalSalesAmount = totalSalesAmount;
        this.totalOrderCount = totalOrderCount;
        this.averageOrderAmount = totalOrderCount > 0 ? 
            totalSalesAmount.divide(BigDecimal.valueOf(totalOrderCount), 2, BigDecimal.ROUND_HALF_UP) : BigDecimal.ZERO;
        this.grossProfit = grossProfit;
        this.profitMargin = profitMargin;
    }

    public void setCategoryBreakdown(String breakdown) {
        this.categoryBreakdown = breakdown;
    }

    public void setProductRanking(String ranking) {
        this.productRanking = ranking;
    }

    public void setChannelBreakdown(String breakdown) {
        this.channelBreakdown = breakdown;
    }

    public Long getId() { return id; }
    public String getReportNo() { return reportNo; }
    public ReportType getType() { return type; }
    public LocalDateTime getReportDate() { return reportDate; }
    public LocalDateTime getStartDate() { return startDate; }
    public LocalDateTime getEndDate() { return endDate; }
    public BigDecimal getTotalSalesAmount() { return totalSalesAmount; }
    public Integer getTotalOrderCount() { return totalOrderCount; }
    public BigDecimal getAverageOrderAmount() { return averageOrderAmount; }
    public BigDecimal getGrossProfit() { return grossProfit; }
    public BigDecimal getProfitMargin() { return profitMargin; }
    public String getCategoryBreakdown() { return categoryBreakdown; }
    public String getProductRanking() { return productRanking; }
    public String getChannelBreakdown() { return channelBreakdown; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setId(Long id) { this.id = id; }
}