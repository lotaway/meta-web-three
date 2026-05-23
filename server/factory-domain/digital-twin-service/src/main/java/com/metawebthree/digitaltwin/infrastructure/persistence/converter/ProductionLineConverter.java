package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.ProductionLineDO;
import java.sql.Timestamp;

public class ProductionLineConverter {

    public static ProductionLineDO toDO(ProductionLine entity) {
        if (entity == null) return null;
        ProductionLineDO d = new ProductionLineDO();
        d.setId(entity.getId());
        d.setLineCode(entity.getLineCode());
        d.setLineName(entity.getLineName());
        d.setWorkshopId(entity.getWorkshopId());
        d.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        d.setCapacity(entity.getCapacity());
        d.setCurrentOutput(entity.getCurrentOutput());
        d.setEfficiency(entity.getEfficiency());
        d.setProductTypes(entity.getProductTypes());
        if (entity.getCreatedAt() != null) d.setCreatedAt(Timestamp.valueOf(entity.getCreatedAt()));
        if (entity.getUpdatedAt() != null) d.setUpdatedAt(Timestamp.valueOf(entity.getUpdatedAt()));
        return d;
    }

    public static ProductionLine toEntity(ProductionLineDO d) {
        if (d == null) return null;
        ProductionLine entity = new ProductionLine();
        entity.setId(d.getId());
        entity.setLineCode(d.getLineCode());
        entity.setLineName(d.getLineName());
        entity.setWorkshopId(d.getWorkshopId());
        if (d.getStatus() != null) entity.setStatus(ProductionLine.ProductionLineStatus.valueOf(d.getStatus()));
        entity.setCapacity(d.getCapacity());
        entity.setCurrentOutput(d.getCurrentOutput());
        entity.setEfficiency(d.getEfficiency());
        entity.setProductTypes(d.getProductTypes());
        if (d.getCreatedAt() != null) entity.setCreatedAt(d.getCreatedAt().toLocalDateTime());
        if (d.getUpdatedAt() != null) entity.setUpdatedAt(d.getUpdatedAt().toLocalDateTime());
        return entity;
    }
}
