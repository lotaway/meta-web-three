package com.metawebthree.reporting.domain.entity;

import java.math.BigDecimal;
import java.math.RoundingMode;
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

    public void setWorkingCapitalMetrics(BigDecimal currentAssets, BigDecimal currentLiabilities) {
        this.currentAssets = currentAssets;
        this.currentLiabilities = currentLiabilities;
        this.workingCapital = currentAssets.subtract(currentLiabilities);
        this.currentRatio = currentLiabilities.compareTo(BigDecimal.ZERO) > 0 ?
            currentAssets.divide(currentLiabilities, 2, RoundingMode.HALF_UP) : BigDecimal.ZERO;
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
    public void setTotalReceivable(BigDecimal totalReceivable) { this.totalReceivable = totalReceivable; }
    public void setTotalPayable(BigDecimal totalPayable) { this.totalPayable = totalPayable; }
    public void setNetReceivable(BigDecimal netReceivable) { this.netReceivable = netReceivable; }
    public void setAgingAnalysis(String agingAnalysis) { this.agingAnalysis = agingAnalysis; }
    public void setCurrentAssets(BigDecimal currentAssets) { this.currentAssets = currentAssets; }
    public void setCurrentLiabilities(BigDecimal currentLiabilities) { this.currentLiabilities = currentLiabilities; }
    public void setWorkingCapital(BigDecimal workingCapital) { this.workingCapital = workingCapital; }
    public void setCurrentRatio(BigDecimal currentRatio) { this.currentRatio = currentRatio; }
    public void setReceivablesByCustomer(String receivablesByCustomer) { this.receivablesByCustomer = receivablesByCustomer; }
    public void setPayablesBySupplier(String payablesBySupplier) { this.payablesBySupplier = payablesBySupplier; }
    public void setReportNo(String reportNo) { this.reportNo = reportNo; }
    public void setType(ReportType type) { this.type = type; }
    public void setReportDate(LocalDateTime reportDate) { this.reportDate = reportDate; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}