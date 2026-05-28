package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.InventoryReport;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.InventoryReportDO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InventoryReportFieldAssigner {
    private static final Logger log = LoggerFactory.getLogger(InventoryReportFieldAssigner.class);

    public static void assignToEntity(InventoryReport report, InventoryReportDO reportDO) {
        report.setId(reportDO.getId());
        report.setReportNo(reportDO.getReportNo());
        report.setType(parseReportType(reportDO.getType()));
        report.setReportDate(reportDO.getReportDate());
        report.setTotalInventoryValue(reportDO.getTotalInventoryValue());
        report.setTotalSkuCount(reportDO.getTotalSkuCount());
        report.setTotalQuantity(reportDO.getTotalQuantity());
        report.setTurnoverRate(reportDO.getTurnoverRate());
        report.setSlowMovingRate(reportDO.getSlowMovingRate());
        report.setSlowMovingCount(reportDO.getSlowMovingCount());
        report.setWarehouseBreakdown(reportDO.getWarehouseBreakdown());
        report.setCategoryBreakdown(reportDO.getCategoryBreakdown());
        report.setLowStockItems(reportDO.getLowStockItems());
        report.setCreatedAt(reportDO.getCreatedAt());
    }

    public static void assignToDO(InventoryReportDO reportDO, InventoryReport report) {
        reportDO.setId(report.getId());
        reportDO.setReportNo(report.getReportNo());
        reportDO.setType(report.getType() != null ? report.getType().name() : null);
        reportDO.setReportDate(report.getReportDate());
        reportDO.setTotalInventoryValue(report.getTotalInventoryValue());
        reportDO.setTotalSkuCount(report.getTotalSkuCount());
        reportDO.setTotalQuantity(report.getTotalQuantity());
        reportDO.setTurnoverRate(report.getTurnoverRate());
        reportDO.setSlowMovingRate(report.getSlowMovingRate());
        reportDO.setSlowMovingCount(report.getSlowMovingCount());
        reportDO.setWarehouseBreakdown(report.getWarehouseBreakdown());
        reportDO.setCategoryBreakdown(report.getCategoryBreakdown());
        reportDO.setLowStockItems(report.getLowStockItems());
        reportDO.setCreatedAt(report.getCreatedAt());
    }

    private static InventoryReport.ReportType parseReportType(String type) {
        if (type == null) {
            return null;
        }
        try {
            return InventoryReport.ReportType.valueOf(type);
        } catch (IllegalArgumentException e) {
            log.warn("Failed to parse InventoryReport.ReportType: invalid value '{}', returning null", type);
            return null;
        }
    }
}