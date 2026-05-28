package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.SalesReport;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.SalesReportDO;
import org.springframework.stereotype.Component;

@Component
public class SalesReportConverter {

    public SalesReport toEntity(SalesReportDO reportDO) {
        if (reportDO == null) {
            return null;
        }
        SalesReport report = new SalesReport();
        SalesReportFieldAssigner.assignToEntity(report, reportDO);
        return report;
    }

    public SalesReportDO toDO(SalesReport report) {
        if (report == null) {
            return null;
        }
        SalesReportDO reportDO = new SalesReportDO();
        SalesReportFieldAssigner.assignToDO(reportDO, report);
        return reportDO;
    }
}