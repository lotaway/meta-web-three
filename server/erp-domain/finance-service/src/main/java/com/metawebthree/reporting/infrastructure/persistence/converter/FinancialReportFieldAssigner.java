package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.FinancialReport;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.FinancialReportDO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FinancialReportFieldAssigner {
    private static final Logger log = LoggerFactory.getLogger(FinancialReportFieldAssigner.class);

    public static void assignToEntity(FinancialReport report, FinancialReportDO reportDO) {
        report.setId(reportDO.getId());
        report.setReportNo(reportDO.getReportNo());
        report.setType(parseReportType(reportDO.getType()));
        report.setReportDate(reportDO.getReportDate());
        report.setTotalReceivable(reportDO.getTotalReceivable());
        report.setTotalPayable(reportDO.getTotalPayable());
        report.setNetReceivable(reportDO.getNetReceivable());
        report.setAgingAnalysis(reportDO.getAgingAnalysis());
        report.setCurrentAssets(reportDO.getCurrentAssets());
        report.setCurrentLiabilities(reportDO.getCurrentLiabilities());
        report.setWorkingCapital(reportDO.getWorkingCapital());
        report.setCurrentRatio(reportDO.getCurrentRatio());
        report.setReceivablesByCustomer(reportDO.getReceivablesByCustomer());
        report.setPayablesBySupplier(reportDO.getPayablesBySupplier());
        report.setCreatedAt(reportDO.getCreatedAt());
    }

    public static void assignToDO(FinancialReportDO reportDO, FinancialReport report) {
        reportDO.setId(report.getId());
        reportDO.setReportNo(report.getReportNo());
        reportDO.setType(report.getType() != null ? report.getType().name() : null);
        reportDO.setReportDate(report.getReportDate());
        reportDO.setTotalReceivable(report.getTotalReceivable());
        reportDO.setTotalPayable(report.getTotalPayable());
        reportDO.setNetReceivable(report.getNetReceivable());
        reportDO.setAgingAnalysis(report.getAgingAnalysis());
        reportDO.setCurrentAssets(report.getCurrentAssets());
        reportDO.setCurrentLiabilities(report.getCurrentLiabilities());
        reportDO.setWorkingCapital(report.getWorkingCapital());
        reportDO.setCurrentRatio(report.getCurrentRatio());
        reportDO.setReceivablesByCustomer(report.getReceivablesByCustomer());
        reportDO.setPayablesBySupplier(report.getPayablesBySupplier());
        reportDO.setCreatedAt(report.getCreatedAt());
    }

    private static FinancialReport.ReportType parseReportType(String type) {
        if (type == null) {
            return null;
        }
        try {
            return FinancialReport.ReportType.valueOf(type);
        } catch (IllegalArgumentException e) {
            log.warn("Failed to parse FinancialReport.ReportType: invalid value '{}', returning null", type);
            return null;
        }
    }
}