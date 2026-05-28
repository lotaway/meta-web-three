package com.metawebthree.reporting.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("rp_sales_report")
public class SalesReportDO {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("report_no")
    private String reportNo;

    @TableField("type")
    private String type;

    @TableField("report_date")
    private LocalDateTime reportDate;

    @TableField("start_date")
    private LocalDateTime startDate;

    @TableField("end_date")
    private LocalDateTime endDate;

    @TableField("total_sales_amount")
    private BigDecimal totalSalesAmount;

    @TableField("total_order_count")
    private Integer totalOrderCount;

    @TableField("average_order_amount")
    private BigDecimal averageOrderAmount;

    @TableField("gross_profit")
    private BigDecimal grossProfit;

    @TableField("profit_margin")
    private BigDecimal profitMargin;

    @TableField("category_breakdown")
    private String categoryBreakdown;

    @TableField("product_ranking")
    private String productRanking;

    @TableField("channel_breakdown")
    private String channelBreakdown;

    @TableField("created_at")
    private LocalDateTime createdAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getReportNo() { return reportNo; }
    public void setReportNo(String reportNo) { this.reportNo = reportNo; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public LocalDateTime getReportDate() { return reportDate; }
    public void setReportDate(LocalDateTime reportDate) { this.reportDate = reportDate; }
    public LocalDateTime getStartDate() { return startDate; }
    public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
    public LocalDateTime getEndDate() { return endDate; }
    public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
    public BigDecimal getTotalSalesAmount() { return totalSalesAmount; }
    public void setTotalSalesAmount(BigDecimal totalSalesAmount) { this.totalSalesAmount = totalSalesAmount; }
    public Integer getTotalOrderCount() { return totalOrderCount; }
    public void setTotalOrderCount(Integer totalOrderCount) { this.totalOrderCount = totalOrderCount; }
    public BigDecimal getAverageOrderAmount() { return averageOrderAmount; }
    public void setAverageOrderAmount(BigDecimal averageOrderAmount) { this.averageOrderAmount = averageOrderAmount; }
    public BigDecimal getGrossProfit() { return grossProfit; }
    public void setGrossProfit(BigDecimal grossProfit) { this.grossProfit = grossProfit; }
    public BigDecimal getProfitMargin() { return profitMargin; }
    public void setProfitMargin(BigDecimal profitMargin) { this.profitMargin = profitMargin; }
    public String getCategoryBreakdown() { return categoryBreakdown; }
    public void setCategoryBreakdown(String categoryBreakdown) { this.categoryBreakdown = categoryBreakdown; }
    public String getProductRanking() { return productRanking; }
    public void setProductRanking(String productRanking) { this.productRanking = productRanking; }
    public String getChannelBreakdown() { return channelBreakdown; }
    public void setChannelBreakdown(String channelBreakdown) { this.channelBreakdown = channelBreakdown; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}