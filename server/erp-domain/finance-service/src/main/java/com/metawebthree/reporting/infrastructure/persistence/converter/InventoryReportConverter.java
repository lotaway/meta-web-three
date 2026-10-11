package com.metawebthree.reporting.infrastructure.persistence.converter;

import com.metawebthree.reporting.domain.entity.InventoryReport;
import com.metawebthree.reporting.infrastructure.persistence.dataobject.InventoryReportDO;
import org.springframework.stereotype.Component;

@Component
public class InventoryReportConverter {

    public InventoryReport toEntity(InventoryReportDO reportDO) {
        if (reportDO == null) {
            return null;
        }
        return InventoryReportFieldAssigner.assignToEntity(reportDO);
    }

    public InventoryReportDO toDO(InventoryReport report) {
        if (report == null) {
            return null;
        }
        InventoryReportDO reportDO = new InventoryReportDO();
        InventoryReportFieldAssigner.assignToDO(reportDO, report);
        return reportDO;
    }
}