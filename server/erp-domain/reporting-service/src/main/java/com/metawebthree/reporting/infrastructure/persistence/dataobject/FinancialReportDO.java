package com.metawebthree.reporting.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@TableName("rp_financial_report")
public class FinancialReportDO {
    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("report_no")
    private String reportNo;

    @TableField("type")
    private String type;

    @TableField("report_date")
    private LocalDateTime reportDate;

    @TableField("total_receivable")
    private BigDecimal totalReceivable;

    @TableField("total_payable")
    private BigDecimal totalPayable;

    @TableField("net_receivable")
    private BigDecimal netReceivable;

    @TableField("aging_analysis")
    private String agingAnalysis;

    @TableField("current_assets")
    private BigDecimal currentAssets;

    @TableField("current_liabilities")
    private BigDecimal currentLiabilities;

    @TableField("working_capital")
    private BigDecimal workingCapital;

    @TableField("current_ratio")
    private BigDecimal currentRatio;

    @TableField("receivables_by_customer")
    private String receivablesByCustomer;

    @TableField("payables_by_supplier")
    private String payablesBySupplier;

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
    public BigDecimal getTotalReceivable() { return totalReceivable; }
    public void setTotalReceivable(BigDecimal totalReceivable) { this.totalReceivable = totalReceivable; }
    public BigDecimal getTotalPayable() { return totalPayable; }
    public void setTotalPayable(BigDecimal totalPayable) { this.totalPayable = totalPayable; }
    public BigDecimal getNetReceivable() { return netReceivable; }
    public void setNetReceivable(BigDecimal netReceivable) { this.netReceivable = netReceivable; }
    public String getAgingAnalysis() { return agingAnalysis; }
    public void setAgingAnalysis(String agingAnalysis) { this.agingAnalysis = agingAnalysis; }
    public BigDecimal getCurrentAssets() { return currentAssets; }
    public void setCurrentAssets(BigDecimal currentAssets) { this.currentAssets = currentAssets; }
    public BigDecimal getCurrentLiabilities() { return currentLiabilities; }
    public void setCurrentLiabilities(BigDecimal currentLiabilities) { this.currentLiabilities = currentLiabilities; }
    public BigDecimal getWorkingCapital() { return workingCapital; }
    public void setWorkingCapital(BigDecimal workingCapital) { this.workingCapital = workingCapital; }
    public BigDecimal getCurrentRatio() { return currentRatio; }
    public void setCurrentRatio(BigDecimal currentRatio) { this.currentRatio = currentRatio; }
    public String getReceivablesByCustomer() { return receivablesByCustomer; }
    public void setReceivablesByCustomer(String receivablesByCustomer) { this.receivablesByCustomer = receivablesByCustomer; }
    public String getPayablesBySupplier() { return payablesBySupplier; }
    public void setPayablesBySupplier(String payablesBySupplier) { this.payablesBySupplier = payablesBySupplier; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}