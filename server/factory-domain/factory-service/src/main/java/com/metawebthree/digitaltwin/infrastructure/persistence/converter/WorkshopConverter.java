package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WorkshopDO;
import java.sql.Timestamp;

public class WorkshopConverter {

    public static WorkshopDO toDO(Workshop entity) {
        if (entity == null) return null;
        WorkshopDO d = new WorkshopDO();
        d.setId(entity.getId());
        d.setWorkshopCode(entity.getWorkshopCode());
        d.setWorkshopName(entity.getWorkshopName());
        d.setDescription(entity.getDescription());
        d.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        d.setArea(entity.getArea());
        d.setLocation(entity.getLocation());
        d.setCenterX(entity.getCenterX());
        d.setCenterY(entity.getCenterY());
        d.setWidth(entity.getWidth());
        d.setLength(entity.getLength());
        if (entity.getCreatedAt() != null) d.setCreatedAt(Timestamp.valueOf(entity.getCreatedAt()));
        if (entity.getUpdatedAt() != null) d.setUpdatedAt(Timestamp.valueOf(entity.getUpdatedAt()));
        return d;
    }

    public static Workshop toEntity(WorkshopDO d) {
        if (d == null) return null;
        Workshop entity = new Workshop();
        entity.setId(d.getId());
        entity.setWorkshopCode(d.getWorkshopCode());
        entity.setWorkshopName(d.getWorkshopName());
        entity.setDescription(d.getDescription());
        if (d.getStatus() != null) entity.setStatus(Workshop.WorkshopStatus.valueOf(d.getStatus()));
        entity.setArea(d.getArea());
        entity.setLocation(d.getLocation());
        entity.setCenterX(d.getCenterX());
        entity.setCenterY(d.getCenterY());
        entity.setWidth(d.getWidth());
        entity.setLength(d.getLength());
        if (d.getCreatedAt() != null) entity.setCreatedAt(d.getCreatedAt().toLocalDateTime());
        if (d.getUpdatedAt() != null) entity.setUpdatedAt(d.getUpdatedAt().toLocalDateTime());
        return entity;
    }
}
