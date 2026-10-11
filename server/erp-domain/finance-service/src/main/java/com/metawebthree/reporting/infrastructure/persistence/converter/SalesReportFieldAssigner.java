package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.SalesReportDO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SalesReportFieldAssigner {
    private static final Logger log = LoggerFactory.getLogger(SalesReportFieldAssigner.class);

    public static void assignToEntity(SalesReport report, SalesReportDO reportDO) {
        report.setId(reportDO.getId());
        report.setReportNo(reportDO.getReportNo());
        report.setType(parseReportType(reportDO.getType()));
        report.setReportDate(reportDO.getReportDate());
        report.setStartDate(reportDO.getStartDate());
        report.setEndDate(reportDO.getEndDate());
        report.setTotalSalesAmount(reportDO.getTotalSalesAmount());
        report.setTotalOrderCount(reportDO.getTotalOrderCount());
        report.setAverageOrderAmount(reportDO.getAverageOrderAmount());
        report.setGrossProfit(reportDO.getGrossProfit());
        report.setProfitMargin(reportDO.getProfitMargin());
        report.setCategoryBreakdown(reportDO.getCategoryBreakdown());
        report.setProductRanking(reportDO.getProductRanking());
        report.setChannelBreakdown(reportDO.getChannelBreakdown());
        report.setCreatedAt(reportDO.getCreatedAt());
    }

    public static void assignToDO(SalesReportDO reportDO, SalesReport report) {
        reportDO.setId(report.getId());
        reportDO.setReportNo(report.getReportNo());
        reportDO.setType(report.getType() != null ? report.getType().name() : null);
        reportDO.setReportDate(report.getReportDate());
        reportDO.setStartDate(report.getStartDate());
        reportDO.setEndDate(report.getEndDate());
        reportDO.setTotalSalesAmount(report.getTotalSalesAmount());
        reportDO.setTotalOrderCount(report.getTotalOrderCount());
        reportDO.setAverageOrderAmount(report.getAverageOrderAmount());
        reportDO.setGrossProfit(report.getGrossProfit());
        reportDO.setProfitMargin(report.getProfitMargin());
        reportDO.setCategoryBreakdown(report.getCategoryBreakdown());
        reportDO.setProductRanking(report.getProductRanking());
        reportDO.setChannelBreakdown(report.getChannelBreakdown());
        reportDO.setCreatedAt(report.getCreatedAt());
    }

    private static SalesReport.ReportType parseReportType(String type) {
        if (type == null) {
            return null;
        }
        try {
            return SalesReport.ReportType.valueOf(type);
        } catch (IllegalArgumentException e) {
            log.warn("Failed to parse SalesReport.ReportType: invalid value '{}', returning null", type);
            return null;
        }
    }
}