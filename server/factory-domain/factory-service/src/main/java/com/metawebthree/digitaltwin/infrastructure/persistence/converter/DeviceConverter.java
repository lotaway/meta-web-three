package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.DeviceDO;
import java.sql.Timestamp;

public class DeviceConverter {

    public static DeviceDO toDO(Device entity) {
        if (entity == null) return null;
        DeviceDO d = new DeviceDO();
        d.setId(entity.getId());
        d.setDeviceCode(entity.getDeviceCode());
        d.setDeviceName(entity.getDeviceName());
        d.setDeviceType(entity.getDeviceType());
        d.setWorkshopId(entity.getWorkshopId());
        d.setProductionLineId(entity.getProductionLineId());
        d.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        d.setPositionX(entity.getPositionX());
        d.setPositionY(entity.getPositionY());
        d.setPositionZ(entity.getPositionZ());
        d.setRotationY(entity.getRotationY());
        d.setIpAddress(entity.getIpAddress());
        d.setMacAddress(entity.getMacAddress());
        d.setMqttTopic(entity.getMqttTopic());
        d.setLastHeartbeat(entity.getLastHeartbeat());
        if (entity.getCreatedAt() != null) d.setCreatedAt(Timestamp.valueOf(entity.getCreatedAt()));
        if (entity.getUpdatedAt() != null) d.setUpdatedAt(Timestamp.valueOf(entity.getUpdatedAt()));
        return d;
    }

    public static Device toEntity(DeviceDO d) {
        if (d == null) return null;
        Device entity = new Device();
        entity.setId(d.getId());
        entity.setDeviceCode(d.getDeviceCode());
        entity.setDeviceName(d.getDeviceName());
        entity.setDeviceType(d.getDeviceType());
        entity.setWorkshopId(d.getWorkshopId());
        entity.setProductionLineId(d.getProductionLineId());
        if (d.getStatus() != null) entity.setStatus(Device.DeviceStatus.valueOf(d.getStatus()));
        entity.setPositionX(d.getPositionX());
        entity.setPositionY(d.getPositionY());
        entity.setPositionZ(d.getPositionZ());
        entity.setRotationY(d.getRotationY());
        entity.setIpAddress(d.getIpAddress());
        entity.setMacAddress(d.getMacAddress());
        entity.setMqttTopic(d.getMqttTopic());
        entity.setLastHeartbeat(d.getLastHeartbeat());
        if (d.getCreatedAt() != null) entity.setCreatedAt(d.getCreatedAt().toLocalDateTime());
        if (d.getUpdatedAt() != null) entity.setUpdatedAt(d.getUpdatedAt().toLocalDateTime());
        return entity;
    }
}
