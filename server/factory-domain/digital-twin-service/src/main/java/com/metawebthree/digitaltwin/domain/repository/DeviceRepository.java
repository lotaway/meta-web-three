package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.Device;
import java.util.List;
import java.util.Optional;

public interface DeviceRepository {
    Optional<Device> findById(Long id);
    Optional<Device> findByDeviceCode(String deviceCode);
    List<Device> findByWorkshopId(String workshopId);
    List<Device> findByProductionLineId(String productionLineId);
    List<Device> findByStatus(Device.DeviceStatus status);
    List<Device> findAll();
    Device save(Device device);
    void update(Device device);
    void deleteById(Long id);
}