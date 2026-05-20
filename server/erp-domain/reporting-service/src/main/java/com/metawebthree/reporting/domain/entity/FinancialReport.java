package com.metawebthree.reporting.domain.entity;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public class FinancialReport {
    private Long id;
    private String reportNo;
    private ReportType type;
    private LocalDateTime reportDate;
    private BigDecimal totalReceivable;
    private BigDecimal totalPayable;
    private BigDecimal netReceivable;
    private String agingAnalysis;
    private BigDecimal currentAssets;
    private BigDecimal currentLiabilities;
    private BigDecimal workingCapital;
    private BigDecimal currentRatio;
    private String receivablesByCustomer;
    private String payablesBySupplier;
    private LocalDateTime createdAt;

    public enum ReportType {
        RECEIVABLE, PAYABLE, AGING, WORKING_CAPITAL
    }

    public void generateReceivableReport(LocalDateTime reportDate) {
        this.reportNo = "AR-" + reportDate.toLocalDate().toString().replace("-", "");
        this.type = ReportType.RECEIVABLE;
        this.reportDate = reportDate;
        this.createdAt = LocalDateTime.now();
    }

    public void generatePayableReport(LocalDateTime reportDate) {
        this.reportNo = "AP-" + reportDate.toLocalDate().toString().replace("-", "");
        this.type = ReportType.PAYABLE;
        this.reportDate = reportDate;
        this.createdAt = LocalDateTime.now();
    }

    public void generateAgingReport(LocalDateTime reportDate) {
        this.reportNo = "AGING-" + reportDate.toLocalDate().toString().replace("-", "");
        this.type = ReportType.AGING;
        this.reportDate = reportDate;
        this.createdAt = LocalDateTime.now();
    }

    public void setReceivableMetrics(BigDecimal totalReceivable, BigDecimal totalPayable) {
        this.totalReceivable = totalReceivable;
        this.totalPayable = totalPayable;
        this.netReceivable = totalReceivable.subtract(totalPayable);
    }

    public void setAgingAnalysis(String analysis) {
        this.agingAnalysis = analysis;
    }

    public void setWorkingCapitalMetrics(BigDecimal currentAssets, BigDecimal currentLiabilities) {
        this.currentAssets = currentAssets;
        this.currentLiabilities = currentLiabilities;
        this.workingCapital = currentAssets.subtract(currentLiabilities);
        this.currentRatio = currentLiabilities.compareTo(BigDecimal.ZERO) > 0 ?
            currentAssets.divide(currentLiabilities, 2, BigDecimal.ROUND_HALF_UP) : BigDecimal.ZERO;
    }

    public void setReceivablesByCustomer(String data) {
        this.receivablesByCustomer = data;
    }

    public void setPayablesBySupplier(String data) {
        this.payablesBySupplier = data;
    }

    public Long getId() { return id; }
    public String getReportNo() { return reportNo; }
    public ReportType getType() { return type; }
    public LocalDateTime getReportDate() { return reportDate; }
    public BigDecimal getTotalReceivable() { return totalReceivable; }
    public BigDecimal getTotalPayable() { return totalPayable; }
    public BigDecimal getNetReceivable() { return netReceivable; }
    public String getAgingAnalysis() { return agingAnalysis; }
    public BigDecimal getCurrentAssets() { return currentAssets; }
    public BigDecimal getCurrentLiabilities() { return currentLiabilities; }
    public BigDecimal getWorkingCapital() { return workingCapital; }
    public BigDecimal getCurrentRatio() { return currentRatio; }
    public String getReceivablesByCustomer() { return receivablesByCustomer; }
    public String getPayablesBySupplier() { return payablesBySupplier; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setId(Long id) { this.id = id; }
}