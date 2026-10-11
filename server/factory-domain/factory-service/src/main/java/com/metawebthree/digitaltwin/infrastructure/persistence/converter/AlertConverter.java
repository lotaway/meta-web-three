package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.AlertDO;
import java.sql.Timestamp;

public class AlertConverter {

    public static AlertDO toDO(Alert entity) {
        if (entity == null) return null;
        AlertDO d = new AlertDO();
        d.setId(entity.getId());
        d.setAlertCode(entity.getAlertCode());
        d.setDeviceCode(entity.getDeviceCode());
        d.setWorkshopId(entity.getWorkshopId());
        d.setLevel(entity.getLevel() != null ? entity.getLevel().name() : null);
        d.setType(entity.getType() != null ? entity.getType().name() : null);
        d.setTitle(entity.getTitle());
        d.setDescription(entity.getDescription());
        d.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        d.setSolution(entity.getSolution());
        d.setAcknowledgedBy(entity.getAcknowledgedBy());
        d.setResolvedBy(entity.getResolvedBy());
        d.setOccurredAt(entity.getOccurredAt());
        d.setAcknowledgedAt(entity.getAcknowledgedAt());
        d.setResolvedAt(entity.getResolvedAt());
        if (entity.getCreatedAt() != null) d.setCreatedAt(Timestamp.valueOf(entity.getCreatedAt()));
        if (entity.getUpdatedAt() != null) d.setUpdatedAt(Timestamp.valueOf(entity.getUpdatedAt()));
        return d;
    }

    public static Alert toEntity(AlertDO d) {
        if (d == null) return null;
        Alert entity = new Alert();
        entity.setId(d.getId());
        entity.setAlertCode(d.getAlertCode());
        entity.setDeviceCode(d.getDeviceCode());
        entity.setWorkshopId(d.getWorkshopId());
        if (d.getLevel() != null) entity.setLevel(Alert.AlertLevel.valueOf(d.getLevel()));
        if (d.getType() != null) entity.setType(Alert.AlertType.valueOf(d.getType()));
        entity.setTitle(d.getTitle());
        entity.setDescription(d.getDescription());
        if (d.getStatus() != null) entity.setStatus(Alert.AlertStatus.valueOf(d.getStatus()));
        entity.setSolution(d.getSolution());
        entity.setAcknowledgedBy(d.getAcknowledgedBy());
        entity.setResolvedBy(d.getResolvedBy());
        entity.setOccurredAt(d.getOccurredAt());
        entity.setAcknowledgedAt(d.getAcknowledgedAt());
        entity.setResolvedAt(d.getResolvedAt());
        if (d.getCreatedAt() != null) entity.setCreatedAt(d.getCreatedAt().toLocalDateTime());
        if (d.getUpdatedAt() != null) entity.setUpdatedAt(d.getUpdatedAt().toLocalDateTime());
        return entity;
    }
}
