package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class DeviceRepositoryImpl implements DeviceRepository {
    private final Map<Long, Device> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<Device> findById(Long id) { return Optional.ofNullable(storage.get(id)); }

    @Override
    public Optional<Device> findByDeviceCode(String code) {
        return storage.values().stream().filter(d -> d.getDeviceCode().equals(code)).findFirst();
    }

    @Override
    public List<Device> findByWorkshopId(String workshopId) {
        return storage.values().stream().filter(d -> d.getWorkshopId().equals(workshopId)).collect(Collectors.toList());
    }

    @Override
    public List<Device> findByProductionLineId(String lineId) {
        return storage.values().stream().filter(d -> d.getProductionLineId().equals(lineId)).collect(Collectors.toList());
    }

    @Override
    public List<Device> findByStatus(Device.DeviceStatus status) {
        return storage.values().stream().filter(d -> d.getStatus() == status).collect(Collectors.toList());
    }

    @Override
    public List<Device> findAll() { return new ArrayList<>(storage.values()); }

    @Override
    public Device save(Device d) { if (d.getId() == null) d.setId(idGen.getAndIncrement()); storage.put(d.getId(), d); return d; }

    @Override
    public void update(Device d) { if (d.getId() != null && storage.containsKey(d.getId())) storage.put(d.getId(), d); }

    @Override
    public void deleteById(Long id) { storage.remove(id); }
}