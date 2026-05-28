package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.FinancialReport;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.FinancialReportDO;
import org.springframework.stereotype.Component;

@Component
public class FinancialReportConverter {

    public FinancialReport toEntity(FinancialReportDO reportDO) {
        if (reportDO == null) {
            return null;
        }
        FinancialReport report = new FinancialReport();
        FinancialReportFieldAssigner.assignToEntity(report, reportDO);
        return report;
    }

    public FinancialReportDO toDO(FinancialReport report) {
        if (report == null) {
            return null;
        }
        FinancialReportDO reportDO = new FinancialReportDO();
        FinancialReportFieldAssigner.assignToDO(reportDO, report);
        return reportDO;
    }
}